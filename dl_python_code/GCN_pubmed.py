import torch
import torch.nn as nn
import gcnconv 
import pubmed_util
import itertools
import torch.nn.functional as F
import pygraph as gone
import kernel
import numpy as np
import datetime
import create_graph as cg

def memoryview_to_np(memview, nebr_dt):
    arr = np.array(memview, copy=False)
    #a = arr.view(nebr_dt).reshape(nebr_reader.get_degree())
    a = arr.view(nebr_dt)
    return a;

if __name__ == "__main__":
    #g_start = datetime.datetime.now()

    ingestion_flag = gone.enumGraph.eDdir | gone.enumGraph.eDoubleEdge 
    ifile = "/home/datalab/data/pubmed/graph_structure"
    num_vcount = 19717
    cuda = torch.device('cuda')
    g_start = datetime.datetime.now()

    G = cg.create_csr_graph_noeid(ifile, num_vcount, ingestion_flag)
    g_end = datetime.datetime.now()
    #print("Graph creation done")
    diff = g_end - g_start
    print ('graph creation time is:', diff)
    
    feature = pubmed_util.read_feature_info("/home/datalab/data/pubmed/feature/feature.txt")
    train_id = pubmed_util.read_index_info("/home/datalab/data/pubmed/index/train_index.txt")
    test_id = pubmed_util.read_index_info("/home/datalab/data/pubmed/index/test_index.txt")
    test_y_label =  pubmed_util.read_label_info("/home/datalab/data/pubmed/label/test_y_label.txt")
    train_y_label =  pubmed_util.read_label_info("/home/datalab/data/pubmed/label/y_label.txt")
    
    feature = torch.tensor(feature)
    train_id = torch.tensor(train_id)
    test_id = torch.tensor(test_id)
    train_y_label = torch.tensor(train_y_label)
    test_y_label = torch.tensor(test_y_label)
    #cuda
    feature=feature.to(cuda)
    train_id=train_id.to(cuda)
    test_id=test_id.to(cuda)
    train_y_label=train_y_label.to(cuda)
    test_y_label=test_y_label.to(cuda)

    # train the network
    input_feature_dim = 500
    net = gcnconv.GCN(G, input_feature_dim, 16, 3)
    net.to(cuda)
    optimizer = torch.optim.Adam(itertools.chain(net.parameters()), lr=0.01, weight_decay=5e-4)
    all_logits = []
    start = datetime.datetime.now()
    for epoch in range(200):
        logits = net(feature)
        #print ('check result')
        #print(logits)
        #print(logits.size())
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        #print("prediction",logp[train_id])
    
        #print('loss_size', logp[train_id].size(), train_y_label.size())
        loss = F.nll_loss(logp[train_id], train_y_label)

        #print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))

        # check the accuracy for test data
        #logits_test = net.forward(feature)
        #logp_test = F.log_softmax(logits_test, 1)

        #acc_val = pubmed_util.accuracy(logp_test[test_id], test_y_label)
        #print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))

    end = datetime.datetime.now()
    difference = end - start
    print("the total training time is:", difference)
    # check the accuracy for test data
    logits_test = net.forward(feature)
    logp_test = F.log_softmax(logits_test, 1)

    acc_val = pubmed_util.accuracy(logp_test[test_id], test_y_label)
    print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))
