import argparse
from networks.model import GNNModel
from util import *
from losses.supcon import SupConLoss
from sklearn.cluster import KMeans
from sklearn import metrics
import torch
import torch.nn.functional as F
from dgl import save_graphs, load_graphs

def parse_argsion():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--device', type=str, default="0",
                        help='id of the used GPU device (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--encoder_name', type=str, default="gat",
                        help='name of the used gnn encoder (default: gat)')

    # model optimization
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch_size for pretrain (default: 2048)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    # dataset and dataset split
    parser.add_argument('--dataset', type=str, default='citeseer', 
                        help='dataset (default: citeseer)')
    parser.add_argument('--nodes_per_class', type=int, default=50,
                        help='number of nodes per seen class in the training/validation set (default: 50)')

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
    parser.add_argument('--tau', type=float, default=0.7,
                        help='temperature of contrastive loss')
    parser.add_argument('--rho', type=int, default=75,
                        help='the ratio for filtering')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='the scaling factor of cross entropy')
    parser.add_argument('--filter', type=int, default=1,
                        help='whether use a threshold to filter pseudo-labeled nodes (0 or 1)')

    args = parser.parse_args()

    return args


def main():
    args = parse_argsion()
    if args.filter != 0 and args.filter != 1:
        raise NotImplementedError(
                'false input of args.filter (0 or 1 is required)')
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    setup_seed(args.seed)
    
    #------------------------------------------load and prepare the dataset-------------------------------------
    g, input_dim, n_class, n_train_class, mask_lab, mask_cls = load_data(args.dataset, args.seed, args.nodes_per_class)
    node_list = list(range(g.num_nodes()))
    train_idx = np.where(g.ndata["train_mask"].numpy() == True)[0].tolist()
    val_test_idx = np.where(g.ndata["val_mask"].numpy() == True)[0].tolist() + np.where(g.ndata["test_mask"].numpy() == True)[0].tolist()
    args.input_dim = input_dim
    args.filter = bool(args.filter)
    args.n_class, args.n_train_class = n_class, n_train_class
    feats = g.ndata["feat"].to(device)
    development_labels = dict()
    for node in train_idx:
        development_labels[node] = g.ndata["label"][node].item()

    #----------------------------------------prepare the model/loss/optimizer-----------------------------------
    model = GNNModel(args).to(device)
    classifier = torch.nn.Linear(args.hidden_dim, args.n_class).to(device)
    criterion = SupConLoss(device, temperature=args.tau).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': classifier.parameters()}], 
                                    lr = args.learning_rate, 
                                    weight_decay=args.weight_decay)
        
    #----------------------------------------------optimize the encoder------------------------------------------
    for epoch in range(1, args.epochs+1):
        model.train()
        classifier.train()
        random.shuffle(node_list)
        for st in range(0, g.num_nodes(), args.batch_size):
            ed = st + args.batch_size
            node_id = node_list[st: ed]
            
            batch_labels = list(range(args.n_class, args.n_class+len(node_id)))
            batch_labels = np.array([development_labels[node] if node in development_labels.keys() else batch_labels[i] for i, node in enumerate(node_id)])
            
            view1 = model(feats, g.to(device))
            view2 = model(feats, g.to(device))
            
            preds1 = classifier(view1)
            preds2 = classifier(view2)
            
            # input_feat = torch.reshape(torch.unsqueeze(torch.cat([view1[node_id], view2[node_id]], dim=1), 1), (-1, 2, args.hidden_dim))
            # input_preds = torch.reshape(torch.unsqueeze(torch.cat([F.normalize(preds1,dim=1)[node_id], F.normalize(preds2,dim=1)[node_id]], dim=1), 1), (-1, 2, args.n_class))
            # input_label = torch.reshape(torch.LongTensor(batch_labels), (-1, 1))
            
            input_feat = torch.cat([view1[node_id].unsqueeze(1), view2[node_id].unsqueeze(1)], dim=1)
            input_preds = torch.cat([F.normalize(preds1,dim=1)[node_id].unsqueeze(1), F.normalize(preds2,dim=1)[node_id].unsqueeze(1)], dim=1)
            input_label = torch.LongTensor(batch_labels)

            if epoch == 1:
                # optimize the encoder with InfoNCE loss during the first training epoch
                loss = criterion(input_feat)
            else:
                loss = criterion(input_feat, input_label) + criterion(input_preds, input_label) + args.scale*torch.nn.CrossEntropyLoss()(preds1[train_idx], g.ndata["label"][train_idx].to(device)) + args.scale*torch.nn.CrossEntropyLoss()(preds2[train_idx], g.ndata["label"][train_idx].to(device))
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # perfrom k-means++ and filter the pseudo-labeled nodes
        model.eval()
        classifier.eval()
        with torch.no_grad():
            emb = model(feats, g.to(device)).detach().cpu().numpy()
            kmeans = KMeans(n_clusters=args.n_class, random_state=args.seed).fit(emb)
            if args.filter:
                centers = kmeans.cluster_centers_
                distance = (np.sum((emb - centers[kmeans.labels_])**2, 1))**0.5
                threshold = np.percentile(distance, args.rho)
                new_labeled_nodes = [idx for idx in val_test_idx if idx in np.where(distance <= threshold)[0]]
            else:
                new_labeled_nodes = val_test_idx
        
        # perform  Hungarian optimal assignment algorithm
        y_pred = kmeans.labels_[mask_lab]
        y_true = g.ndata["label"].numpy()[mask_lab]
        D = args.n_class
        w = np.zeros((D, D), dtype=int)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T
        ind_map = {i: j for i, j in ind}

        if len(set(ind[:, 0])) != args.n_class:
            break
        if len(set(ind[:, 1])) != args.n_class:
            break

        # pseudo-labeling
        if len(new_labeled_nodes) > 0:
            print("Adding {} pseudo-labeled nodes ...".format(len(new_labeled_nodes)))
            # development_labels = dict()
            # for node in range(g.num_nodes()):
            #     if node in train_idx:
            #         development_labels[node] = g.ndata["label"][node].item()   
            #     if (node in new_labeled_nodes) and (node in val_test_idx):
            #         development_labels[node] = ind_map[kmeans.labels_[node]]
            nodes_pl = list(np.intersect1d(np.array(new_labeled_nodes),np.array(val_test_idx)))
            labels_pl = kmeans.labels_[nodes_pl].tolist()
            labels_pl = np.array([ind_map[elem] for elem in labels_pl])
            development_labels = dict(map(lambda x,y:[x,y], nodes_pl, labels_pl))
            for node in train_idx:
                development_labels[node] = g.ndata["label"][node].item()
        
        # prediction and evaluation
        targets = g.ndata["label"].numpy()
        preds = kmeans.labels_
        preds = np.array([ind_map[elem] for elem in preds]) # alignment
        if len(set(preds[val_test_idx])) == 1:
            score = -1
        else:
            score = metrics.silhouette_score(emb[val_test_idx], preds[val_test_idx])
        
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)
        mask2 = mask_cls[g.ndata["val_mask"]]
        mask2 = mask2.astype(bool)
        all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets[~mask_lab], y_pred=preds[~mask_lab], mask=mask)
        val_acc, _, _ = split_cluster_acc_v2(y_true=targets[g.ndata["val_mask"]], y_pred=preds[g.ndata["val_mask"]], mask=mask2)
        print(("val_acc: {:.5f}, all_acc: {:.5f}, old_acc: {:.5f}, new_acc: {:.5f}\n".format(val_acc, all_acc, old_acc, new_acc)))
        
        # compute imbalance rate and separation rate
        seen, novel = [], []
        for i in range(args.n_train_class):
            seen.extend(list(np.where(targets==i)[0]))
        seen_variancs, seen_centers = compute_var_mean(emb[seen], targets[seen])

        for i in range(args.n_train_class, args.n_class):
            novel.extend(list(np.where(targets==i)[0]))
        novel_variancs, novel_centers = compute_var_mean(emb[novel], targets[novel])

        imbalance_rate = []
        separate_rate = []
        for var1, center1 in zip(novel_variancs, novel_centers):
            for var2, center2 in zip(seen_variancs, seen_centers):
                imbalance_rate.append(var1 / var2)
                separate_rate.append(np.sum((center1 - center2)**2)**0.5 / (var1 + var2))

        # write down the evaluation results
        fp=open("./log/{}/res_ours_{}_{}.log".format(args.dataset, args.dataset, args.seed), "a")
        fp.write("alpha: {}, tau: {}, labeled_nodes_per_seen_class: {}, lr: {}, scale:{}, pretrain_epochs: {}, score:{:.5f}, val_acc: {:.5f}, all_acc: {:.5f}, old_acc: {:.5f}, new_acc: {:.5f}, imbalance_rate_mean: {:.5f}, separate_rate_mean: {:.5f}, imbalance_rate_min: {:.5f}, separate_rate_min: {:.5f}\n".format(args.rho, args.tau, args.nodes_per_class, args.learning_rate, args.scale, epoch, score, val_acc, all_acc, old_acc, new_acc, np.mean(imbalance_rate), np.mean(separate_rate), np.min(imbalance_rate), np.min(separate_rate)))
        fp.close()


if __name__ == '__main__':
    main()
