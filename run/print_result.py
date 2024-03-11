import sys

def extract_result(file_path, weight):
    f = open(file_path,"r")
    lines = f.readlines()
    
    best, max_score, max_acc, min_score, min_acc = 0, 0, 0, 100000, 100000
    for index in range(len(lines)):
        line = lines[index]
        score = float(line.split("score:")[1].split(",")[0].strip())
        val_acc = float(line.split("val_acc:")[1].split(",")[0].strip())
        
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
        if val_acc > max_acc:
            max_acc = val_acc
        if val_acc < min_acc:
            min_acc = val_acc

    for index in range(len(lines)):
        line = lines[index]
        score = float(line.split("score:")[1].split(",")[0].strip())
        val_acc = float(line.split("val_acc:")[1].split(",")[0].strip())
        all_acc = float(line.split("all_acc:")[1].split(",")[0].strip())
        old_acc = float(line.split("old_acc:")[1].split(",")[0].strip())
        new_acc = float(line.split("new_acc:")[1].split(",")[0].strip())
        imbalance_rate = float(line.split("imbalance_rate_mean:")[1].split(",")[0].strip())
        separate_rate = float(line.split("separate_rate_mean:")[1].split(",")[0].strip())

        final_score = weight * (score - min_score) / (max_score - min_score) + (1-weight) * (val_acc - min_acc) / (max_acc - min_acc)
    
        if final_score >= best:
            best = final_score
            best_line = line
            best_index = index
            best_all_acc = all_acc
            best_old_acc = old_acc
            best_new_acc = new_acc
            best_imbalance_rate = imbalance_rate
            best_separate_rate = separate_rate
    # if index != 20-1:
    #     print("warning!")
    # print(index)
    f.close()
    return best, best_line, best_index, best_all_acc, best_old_acc, best_new_acc, best_imbalance_rate, best_separate_rate

def extract_result_large(file_path, weight):
    f = open(file_path,"r")
    lines = f.readlines()
    
    best, max_score, max_acc, min_score, min_acc = 0, 0, 0, 100000, 100000
    for index in range(len(lines)):
        line = lines[index]
        score = float(line.split("score:")[1].split(",")[0].strip())
        val_acc = float(line.split("val_acc:")[1].split(",")[0].strip())
        
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
        if val_acc > max_acc:
            max_acc = val_acc
        if val_acc < min_acc:
            min_acc = val_acc

    for index in range(len(lines)):
        line = lines[index]
        score = float(line.split("score:")[1].split(",")[0].strip())
        val_acc = float(line.split("val_acc:")[1].split(",")[0].strip())
        all_acc = float(line.split("all_acc:")[1].split(",")[0].strip())
        old_acc = float(line.split("old_acc:")[1].split(",")[0].strip())
        new_acc = float(line.split("new_acc:")[1].split(",")[0].strip())
        
        final_score = weight * (score - min_score) / (max_score - min_score) + (1-weight) * (val_acc - min_acc) / (max_acc - min_acc)
    
        if final_score >= best:
            best = final_score
            best_line = line
            best_index = index
            best_all_acc = all_acc
            best_old_acc = old_acc
            best_new_acc = new_acc
    # if index != 20-1:
    #     print("warning!")
    # print(index)
    f.close()
    return best, best_line, best_index, best_all_acc, best_old_acc, best_new_acc


dataset = sys.argv[1]
weight = float(sys.argv[2])
num=int(sys.argv[3])
seeds = [2406525885, 1660347731, 3164031153, 1454191016, 1583215992, 765984986, 258270452, 3808600642, 292690791, 2492579272]
seeds = seeds[:num]

print("==================OpenIMR==================")
avg_all = 0
avg_old = 0
avg_new = 0

if dataset != "ogbn-arxiv" and dataset != "ogbn-products":
    avg_imbalance_rate = 0
    avg_separate_rate = 0
    for seed in seeds:
        best, best_line, best_index, best_all_acc, best_old_acc, best_new_acc, best_imbalance_rate, best_separate_rate = extract_result(f"../log/"+dataset+"/res_ours_"+dataset+"_"+str(seed)+".log", weight=weight)
        avg_all += best_all_acc
        avg_old += best_old_acc
        avg_new += best_new_acc
        avg_imbalance_rate += best_imbalance_rate
        avg_separate_rate += best_separate_rate
    print(("Overall_Acc (OpenIMR): {} | Seen_Acc: {} | Novel_Acc: {} | Imbalance rate (Avg.): {} | Separation rate (Avg.): {}").format(avg_all / num, avg_old / num, avg_new / num, avg_imbalance_rate / num, avg_separate_rate / num))

else:
    for seed in seeds:
        best, best_line, best_index, best_all_acc, best_old_acc, best_new_acc = extract_result_large(f"../log/"+dataset+"/res_ours_"+dataset+"_"+str(seed)+".log", weight=weight)
        avg_all += best_all_acc
        avg_old += best_old_acc
        avg_new += best_new_acc
    print(("Overall_Acc (OpenIMR): {} | Seen_Acc: {} | Novel_Acc: {}").format(avg_all / num, avg_old / num, avg_new / num))

