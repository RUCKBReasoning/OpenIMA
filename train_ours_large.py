import argparse
import torch
from networks_large.model import GNNModel
from util import *
from losses.supcon import SupConLoss
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from dgl.dataloading import MultiLayerFullNeighborSampler, NeighborSampler, NodeDataLoader

import warnings
warnings.filterwarnings("ignore")

def parse_argsion():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--device', type=str, default="0",
                        help='id of the used GPU device (default: 0)')
    parser.add_argument('--seed', type=int, default="0",
                        help='random seed (default: 0)')
    parser.add_argument('--encoder_name', type=str, default="gat",
                        help='name of the used gnn encoder (default: gat)')

    # model optimization
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch_size for pretrain (default: 2048)')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help='number of training epochs of supcon (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    # dataset and dataset split
    parser.add_argument('--dataset', type=str, default='citeseer', 
                        help='dataset (default: citeseer)')
    parser.add_argument('--nodes_per_class', type=int, default=500,
                        help='number of nodes per class in the training set (default: 500)')

    # gnn hyper-parameters
    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help='number of layers in graphcnn (not include the input layer)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='dimensionality of hidden units (default: 128)')
    parser.add_argument('--num_gnn_heads', type=int, default=8,
                        help='the number of heads in the GAT layer (default: 8)')
    parser.add_argument('--feat_drop_rate', type=float, default=0.5,
                        help='dropout rate on feature (default: 0.5)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.5,
                        help='dropout rate on attention weight for GAT layer (default: 0.5)')

    # loss hyper-parameters
    parser.add_argument('--tau', type=float, default=0.07,
                        help='temperature for supcon loss function')
    parser.add_argument('--alpha', type=int, default=75,
                        help='the ratio for filtering')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='the weight of cross entropy')
    parser.add_argument('--filter', type=int, default=1,
                        help='whether use threshold to filter new labeled nodes')

    args = parser.parse_args()

    return args


def main():
    args = parse_argsion()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    setup_seed(args.seed)
    
    #------------------------------------------load and prepare the dataset-------------------------------------
    g, input_dim, n_class, n_train_class, mask_lab, mask_cls = load_data_ogb(args.dataset, args.seed, args.nodes_per_class)
    train_idx = np.where(g.ndata["train_mask"].numpy() == True)[0].tolist()
    val_test_idx = np.where(g.ndata["val_mask"].numpy() == True)[0].tolist() + np.where(g.ndata["test_mask"].numpy() == True)[0].tolist()
    args.input_dim = input_dim
    args.filter = bool(args.filter)
    args.n_class, args.n_train_class = n_class, n_train_class
    development_labels = dict()
    for node in train_idx:
        development_labels[node] = g.ndata["label"][node].item()

    #----------------------------------------prepare the model/loss/optimizer-----------------------------------
    model = GNNModel(args).to(device)
    classifier = torch.nn.Linear(args.hidden_dim, args.n_class).to(device)
    criterion = SupConLoss(device, temperature=args.tau)
    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': classifier.parameters()}], 
                                    lr = args.learning_rate, 
                                    weight_decay=args.weight_decay)
                                 
    #----------------------------------optimize the encoder in a self-training way-----------------------------
    sampler = NeighborSampler([10, 20])
    dataloader = NodeDataLoader(g, torch.tensor(np.arange(g.num_nodes()), dtype=torch.int64), sampler, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(1, args.pretrain_epochs+1):
        model.train()
        classifier.train()
        for _, output_nodes, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]
            node_id = output_nodes.numpy()
            batch_labels = list(range(args.n_class, args.n_class+len(node_id)))
            batch_labels = np.array([development_labels[node] if node in development_labels.keys() else batch_labels[i] for i, node in enumerate(node_id)])
            view1 = model(blocks[0].srcdata['feat'], blocks)
            view2 = model(blocks[0].srcdata['feat'], blocks)
            preds1 = classifier(view1)
            preds2 = classifier(view2)
            
            dataloader_batchlbl = NodeDataLoader(g, torch.tensor(np.array(train_idx), dtype=torch.int64), sampler, batch_size=len(train_idx), shuffle=False)
            _, output_nodes_lbl, blocks_lbl = next(iter(dataloader_batchlbl))
            temp_labels = g.ndata["label"][output_nodes_lbl]
            blocks_lbl = [block.to(device) for block in blocks_lbl]
            view3 = model(blocks_lbl[0].srcdata['feat'], blocks_lbl)
            view4 = model(blocks_lbl[0].srcdata['feat'], blocks_lbl)
            preds3 = classifier(view3)
            preds4 = classifier(view4)
            
            input_feat = torch.reshape(torch.unsqueeze(torch.cat([view1, view2], dim=1), 1), (-1, 2, args.hidden_dim))
            input_preds = torch.reshape(torch.unsqueeze(torch.cat([F.normalize(preds1,dim=1), F.normalize(preds2,dim=1)], dim=1), 1), (-1, 2, args.n_class))
            input_label = torch.reshape(torch.LongTensor(batch_labels), (-1, 1))

            # pairwise loss
            emb_detach = view1.detach()
            emb_norm = emb_detach / torch.norm(emb_detach, 2, 1, keepdim=True)
            cosine_dist = torch.mm(emb_norm, emb_norm.t())

            pos_pairs = []
            # unlabel part
            unlabel_cosine_dist = cosine_dist[:, :]
            vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
            pos_pairs.extend(pos_idx)
            prob = F.softmax(preds1, dim=1)
            prob2 = F.softmax(preds2, dim=1)
            pos_prob = prob2[pos_pairs, :]
            bz = len(output_nodes)
            pos_sim = torch.bmm(prob.view(bz, 1, -1), pos_prob.view(bz, -1, 1)).squeeze()
            
            ones = torch.ones_like(pos_sim)
            bce_loss = torch.nn.BCELoss()(pos_sim, ones)
            
            if epoch == 1:
                loss_supcon = criterion(input_feat)
            else:
                loss_supcon = bce_loss + criterion(input_feat, input_label) + criterion(input_preds, input_label) + args.scale*torch.nn.CrossEntropyLoss()(preds3, torch.tensor(np.array(temp_labels), dtype=torch.int64).to(device)) + args.scale*torch.nn.CrossEntropyLoss()(preds4, torch.tensor(np.array(temp_labels), dtype=torch.int64).to(device))
            if optimizer is not None:
                optimizer.zero_grad()
                loss_supcon.backward()
                optimizer.step()
    
        # perfrom k-means and filter the new labeled nodes
        model.eval()
        classifier.eval()
        dataloader_test = NodeDataLoader(g, torch.tensor(np.arange(g.num_nodes()), dtype=torch.int64), sampler, batch_size=args.batch_size, shuffle=False)
        emb, preds_all, confs_all = [], [], []
        for _, _, blocks_test in dataloader_test:
            blocks_test = [block.to(device) for block in blocks_test]
            with torch.no_grad():
                view = model(blocks_test[0].srcdata['feat'], blocks_test)
                output = classifier(view).detach()
                prob = F.softmax(output, dim=1)
                confs, preds_temp = prob.max(1)
            emb.append(view.detach().cpu())
            preds_all.append(preds_temp.cpu())
            confs_all.append(confs.cpu())
        emb = torch.cat(emb, dim=0).numpy()
        preds_all = torch.cat(preds_all).numpy()
        confs_all = torch.cat(confs_all).numpy()
        
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=args.n_class, batch_size=2048000,
                    n_init=10, max_no_improvement=10, verbose=0, random_state=args.seed).fit(emb)
        if args.filter:
            centers = kmeans.cluster_centers_
            distance = (np.sum((emb - centers[kmeans.labels_])**2, 1))**0.5
            threshold = np.percentile(distance, args.alpha)
            new_labeled_nodes = [idx for idx in val_test_idx if idx in np.where(distance <= threshold)[0]]
        else:
            new_labeled_nodes = val_test_idx
        
        # alignment
        y_pred = kmeans.labels_[mask_lab]
        y_true = g.ndata["label"].numpy()[mask_lab]
        D = args.n_class
        w = np.zeros((D, D), dtype=int)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T
        ind_map = {i: j for i, j in ind}
        
        if len(new_labeled_nodes) > 0:
            print("Adding {} pseudo-labeled nodes ...".format(len(new_labeled_nodes)))
            nodes_pl = list(np.intersect1d(np.array(new_labeled_nodes),np.array(val_test_idx)))
            labels_pl = kmeans.labels_[nodes_pl].tolist()
            labels_pl = np.array([ind_map[elem] for elem in labels_pl])
            development_labels = dict(map(lambda x,y:[x,y], nodes_pl, labels_pl))
            for node in train_idx:
                development_labels[node] = g.ndata["label"][node].item()

        print("Evaluation ...")
        targets = g.ndata["label"].numpy()
        choice = np.random.choice(list(range(len(val_test_idx))), size=10000, replace=False, p=None)
        preds = preds_all

        print("Compute SC ...")
        if len(set(preds[val_test_idx][choice])) == 1:
            score = -1
        else:
            score = metrics.silhouette_score(emb[val_test_idx][choice], preds[val_test_idx][choice])
        
        print("Compute ACC ...")
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)
        mask2 = mask_cls[g.ndata["val_mask"]]
        mask2 = mask2.astype(bool)
        all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets[~mask_lab], y_pred=preds[~mask_lab], mask=mask)
        val_acc, _, _ = split_cluster_acc_v2(y_true=targets[g.ndata["val_mask"]], y_pred=preds[g.ndata["val_mask"]], mask=mask2)
        print(("val_acc: {:.5f}, all_acc: {:.5f}, old_acc: {:.5f}, new_acc: {:.5f}\n".format(val_acc, all_acc, old_acc, new_acc)))
        
        # write down the evaluation results
        fp=open("./log/{}/res_ours_{}_{}.log".format(args.dataset, args.dataset, args.seed), "a")
        fp.write("alpha: {}, tau: {}, lr: {}, scale:{}, pretrain_epochs: {}, score:{:.5f}, val_acc: {:.5f}, all_acc: {:.5f}, old_acc: {:.5f}, new_acc: {:.5f}\n".format(args.alpha, args.tau, args.learning_rate, args.scale, epoch, score, val_acc, all_acc, old_acc, new_acc))
        fp.close()

if __name__ == '__main__':
    main()
