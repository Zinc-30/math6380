//
//  rank.cpp
//  xcodeProject1
//
//  Created by Jinxing on 13/5/2017.
//  Copyright © 2017 Jinxing. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;
#define MAXSTRING 100
#define CELL_SIZE 990
map<string, int> drug_dict;
map<string, int> cell_dict;
typedef double real;
real lr = 1e-3;
real weight_scale = 1e-3;
int batch_size = 1000;
int epoch = 1000;
int drug_size = 265, gene_size = 1250, class_size = 3;
vector<int> cell_feature[CELL_SIZE];
int train_data_nums = 0;
int test_data_nums = 0;
int print_epo = 1;
/*
 Model parameters:
 drugW, the embedding matrix of drugs
 drugb, the bias vector of drugs
 affineW, affineb, the affine layer, convert the rank scores to softmax scores
 cell_embed, temp vector to store the embedding of cell
 */
real *drugW, *drugb, *affineW, *affineb;
// gradients of parameters
real *grad_drugW, *grad_drugb, *grad_affineW, *grad_affineb;
struct DataPair{
    int sample_id;
    int cell_id;
    int drug1_id;
    int drug2_id;
    int y;
};
vector<DataPair> train_pairs;
vector<DataPair> test_pairs;
void ReadDrugDict(char* filename)
{
    ifstream infile(filename);
    if (infile.is_open()){
        string tmp_drug;
        int idx = 0;
        while (infile >> tmp_drug){
            drug_dict[tmp_drug] = idx;
            idx++;
        }
        infile.close();
    }
    else{
        printf("ERROR: failed to read the file in ReadDrugDict\n");
        exit(1);
    }
}
void ReadCellDict(char* filename)
{
    ifstream infile(filename);
    if (infile.is_open()){
        string tmp_cell;
        int idx = 0;
        while (infile >> tmp_cell){
            cell_dict[tmp_cell] = idx;
            idx++;
        }
        infile.close();
    }
    else{
        printf("ERROR: failed to read the file in ReadDrugDict\n");
        exit(1);
    }
}
// Read the features of the drug
void ReadCellFeature(char* filename)
{
    ifstream infile(filename);
    if (infile.is_open()){
        int cnt = 0;
        string line;
        while(getline(infile,line)){
            istringstream iss(line);
            int tmp;
            while(iss >> tmp)
                cell_feature[cnt].push_back(tmp);
            cnt++;
        }
        infile.close();
    }
    else{
        printf("ERROR: failed to read the file of cell features\n");
        exit(1);
    }
}
// Load the training pairs
void ReadTrainPairs(char* filename)
{
    ifstream infile(filename);
    if (infile.is_open()){
        string line;
        while (getline(infile, line)){
            istringstream iss(line);
            DataPair tmp;
            iss >> tmp.sample_id;
            tmp.sample_id += 1;
            string cell_name, drug1_name, drug2_name;
            iss >> cell_name;
            iss >> drug1_name;
            iss >> drug2_name;
            tmp.cell_id = cell_dict[cell_name];
            tmp.drug1_id = drug_dict[drug1_name];
            tmp.drug2_id = drug_dict[drug2_name];
            float y = 0;
            iss >> y;
            tmp.y = int(y);
            train_pairs.push_back(tmp);
        }
        infile.close();
        train_data_nums = (int) train_pairs.size();
    }
    else{
        printf("ERROR: failed to read the file of training pairs\n");
        exit(1);
    }
}
// Load the testing pairs
void ReadTestPairs(char* filename)
{
    ifstream infile(filename);
    if (infile.is_open()){
        string line;
        while (getline(infile, line)){
            istringstream iss(line);
            DataPair tmp;
            tmp.y = 0;  // intialize the unknow value to 0
            iss >> tmp.sample_id;
            tmp.sample_id += 1;
            string cell_name, drug1_name, drug2_name;
            iss >> cell_name;
            iss >> drug1_name;
            iss >> drug2_name;
            tmp.cell_id = cell_dict[cell_name];
            tmp.drug1_id = drug_dict[drug1_name];
            tmp.drug2_id = drug_dict[drug2_name];
            test_pairs.push_back(tmp);
        }
        infile.close();
        test_data_nums = int(test_pairs.size());
    }
    else{
        printf("ERROR: failed to read the file of testing pairs\n");
        exit(1);
    }
}

void ClearGradients()
{
    for (int i = 0; i < drug_size * gene_size; i++) grad_drugW[i] = 0;
    for (int i = 0; i < drug_size; i++) grad_drugb[i] = 0;
    for (int i = 0; i < class_size; i++) grad_affineW[i] = 0;
    for (int i = 0; i < class_size; i++) grad_affineb[i] = 0;
}

void UpdateParameters()
{
    for (int i = 0; i < drug_size * gene_size; i++) drugW[i] += - lr * grad_drugW[i];
    for (int i = 0; i < drug_size; i++) drugb[i] += -lr * grad_drugb[i];
    for (int i = 0; i < class_size; i++) affineW[i] += -lr * grad_affineW[i];
    for (int i = 0; i < class_size; i++) affineb[i] += -lr * grad_affineb[i];
}

struct Result{
    int pred_y;
    real loss;
};

// Predict the drug sensitivity rank for each data sample
// Only backpropagate the gradients during the training period
Result TrainPredictOneData(DataPair cur_pair, bool train)
{
    Result result;
    // forward pass
    int tmp_cell_id = cur_pair.cell_id;

    int tmp_gene_num = (int) cell_feature[tmp_cell_id].size();
    int tmp_drug1_id = cur_pair.drug1_id;
    int tmp_drug2_id = cur_pair.drug2_id;
    // computhe the score s = b1 - b2 + <x, beta_1 - beta_2>
    real s = drugb[tmp_drug1_id] - drugb[tmp_drug2_id];
    for (int k = 0; k < tmp_gene_num; k++){
        int gene_id = cell_feature[tmp_cell_id][k];
        s += drugW[tmp_drug1_id * gene_size + gene_id] - drugW[tmp_drug2_id * gene_size + gene_id];
    }
    
    // transform the score s to target labels probability by affine - softmax layers
    real prob[3];
    real maxp = 0;
    int maxid = 0;
    for (int k = 0; k < 3; k++){
        prob[k] = s * affineW[k] + affineb[k];
        if (prob[k] > maxp){
            maxp = prob[k];
            maxid = k;
        }
    }
    
    // transform by softmax function
    real sum = 0;
    for (int k = 0; k < 3; k++){
        prob[k] -= maxp;
        prob[k] = exp(prob[k]);
        sum += prob[k];
    }
    
    for (int k = 0; k < 3; k++)
        prob[k] /= sum;
    result.pred_y = maxid;
    result.loss = 0;
    if (train){
        int tmp_y = cur_pair.y;
        result.loss = -log(prob[tmp_y+1]);
        // backpropagate the gradients
        real dx[3];
        for (int k = 0; k < 3; k++)
            dx[k] = prob[k];
        dx[tmp_y+1] -= 1;
        for (int k = 0; k < 3; k++){
            grad_affineW[k] += dx[k] * s;
            grad_affineb[k] += dx[k];
        }
        real ds = 0;
        for (int k = 0; k < 3; k++)
            ds += dx[k] * affineW[k];

        grad_drugb[tmp_drug1_id] += ds;
        grad_drugb[tmp_drug2_id] += -ds;
        for (int k = 0; k < tmp_gene_num; k++){
            int gene_id = cell_feature[tmp_cell_id][k];
            grad_drugW[tmp_drug1_id * gene_size + gene_id] += ds;
            grad_drugW[tmp_drug2_id * gene_size + gene_id] += -ds;
        }
    }
    return result;
}
// partition the training data into training set and validation set
void Train()
{
    // random shuffle the training data
    random_shuffle(train_pairs.begin(), train_pairs.end());
    int sample_train_size = train_data_nums / 5 * 4;
    real loss = 0, train_acc = 0, val_acc = 0;
    for (int iter = 0; iter < epoch; iter++){
        // train and update parameters
        for (int i = 0; i < sample_train_size; i = i + batch_size){
            real batch_loss = 0;
            ClearGradients();
            for (int j = i; j < i + batch_size && j < sample_train_size; j++){
                Result result = TrainPredictOneData(train_pairs[j], true);
                batch_loss += result.loss;
            }
            // printf("batch loss %.4f\n",batch_loss);
            // update parameters by SGD
            UpdateParameters();
            
            loss += batch_loss;
        }
        loss /= sample_train_size / batch_size;
        // train accuracy
        for (int i = 0; i < sample_train_size; i++){
        	Result result = TrainPredictOneData(train_pairs[i], false);
        	if (result.pred_y == train_pairs[i].y + 1)
        		train_acc += 1.0 / sample_train_size;
        }
        // validation accuracy
        for (int i = sample_train_size; i < train_data_nums; i++){
            Result result = TrainPredictOneData(train_pairs[i], false);
            if (result.pred_y == train_pairs[i].y + 1)
                val_acc += 1.0 / (train_data_nums - sample_train_size);
        }
        // print average loos and training accuracy
        if ((iter % print_epo) == 0){
            printf("==========> epoch %d  loss %.4f train-acc %.4f val-acc %.4f\n", iter, loss / print_epo, train_acc / print_epo, val_acc / print_epo);
            loss = 0;
            train_acc = 0;
            val_acc = 0;
        }
    }
}
// use the whole dataset for training
void Train2()
{
    // random shuffle the training data
    random_shuffle(train_pairs.begin(), train_pairs.end());
    real loss = 0, train_acc = 0;
    for (int iter = 0; iter < epoch; iter++){
        // train and update parameters
        for (int i = 0; i < train_data_nums; i = i + batch_size){
            real batch_loss = 0;
            ClearGradients();
            for (int j = i; j < i + batch_size && j < train_data_nums; j++){
                Result result = TrainPredictOneData(train_pairs[j], true);
                batch_loss += result.loss;
            }
            // printf("batch loss %.4f\n",batch_loss);
            // update parameters by SGD
            UpdateParameters();
            
            loss += batch_loss;
        }
        loss /= train_data_nums / batch_size;
        // train accuracy
        for (int i = 0; i < train_data_nums; i++){
            Result result = TrainPredictOneData(train_pairs[i], false);
            if (result.pred_y == train_pairs[i].y + 1)
                train_acc += 1.0 / train_data_nums;
        }
        // print average loos and training accuracy
        if ((iter % print_epo) == 0){
            printf("==========> epoch %d  loss %.4f train-acc %.4f\n", iter, loss / print_epo, train_acc / print_epo);
            loss = 0;
            train_acc = 0;
        }
    }
}
void Test(char *test_predict_file)
{
    // predict the drug pairs in test file
    ofstream ofile(test_predict_file);
    ofile << "SampleID, ComparisonValue" << endl;
    for (int i = 0; i < test_data_nums; i++){
        Result result = TrainPredictOneData(test_pairs[i], false);
        test_pairs[i].y = result.pred_y;
        ofile << test_pairs[i].sample_id << "," << test_pairs[i].y - 1 << endl;
    }
    ofile.close();
}
void normal_initialize(real *array, int asize)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (0.0,1.0);
    for (int i = 0; i < asize; i++)
        array[i] = weight_scale * distribution(generator);
}
void SaveParameters(char* parameter_file)
{
    ofstream ofile(parameter_file);
    ofile << "drugW" << endl;
    ofile << drugW[0];
    int i = 0;
    for (i = 1; i < drug_size * gene_size; i++)
        ofile << " " << drugW[i];
    ofile << endl;
    ofile << "drugb" << endl;
    ofile << drugb[0];
    for (i = 1; i < drug_size; i++)
        ofile << " " << drugb[i];
    ofile << endl;
    ofile << "affineW" << endl;
    ofile << affineW[0];
    for (i = 1; i < class_size; i++)
        ofile << " " << affineW[i];
    ofile << endl;
    ofile << "affineb" << endl;
    ofile << affineb[0];
    for (i = 1; i < class_size; i++)
        ofile << " " << affineb[i];
    ofile << endl;
    ofile.close();
}
void ReadParameters(char *parameter_file)
{
    ifstream infile(parameter_file);
    if (infile.is_open()){
        string name;
        string line;
        int i = 0;
        getline(infile,name);
        getline(infile,line);
        istringstream iss(line);
        for (i = 0; i < drug_size * gene_size; i++)
            iss >> drugW[i];
        getline(infile,name);
        getline(infile,line);
        iss.str(line);
        for (i = 0; i < drug_size; i++)
            iss >> drugb[i];
        getline(infile,name);
        getline(infile,line);
        iss.str(line);
        for (i = 0; i < class_size; i++)
            iss >> affineW[i];
        getline(infile,name);
        getline(infile,line);
        iss.str(line);
        for (i = 0; i < class_size; i++)
            iss >> affineb[i];
        infile.close();
    }
    else{
        printf("ERROR: failed to read the parameter file\n");
        exit(1);
    }
}
void InitParameters()
{
    drugW = (real *)calloc(drug_size * gene_size, sizeof(real));
    drugb = (real *)calloc(drug_size, sizeof(real));
    affineW = (real *)calloc(class_size, sizeof(real));
    affineb = (real *)calloc(class_size, sizeof(real));

    normal_initialize(drugW, drug_size * gene_size);
    normal_initialize(affineW, class_size);

    grad_drugW = (real *)calloc(drug_size * gene_size, sizeof(real));
    grad_drugb = (real *)calloc(drug_size, sizeof(real));
    grad_affineW = (real *)calloc(class_size, sizeof(real));
    grad_affineb = (real *)calloc(class_size, sizeof(real));
}
void DestroyParameters()
{
    free(drugW);
    free(drugb);
    free(affineW);
    free(affineb);

    free(grad_drugW);
    free(grad_drugb);
    free(grad_affineW);
    free(grad_affineb);
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char**argv)
{
    char drug_list_file[MAXSTRING], cell_list_file[MAXSTRING], cell_feature_file[MAXSTRING];
    char train_pairs_file[MAXSTRING], test_pairs_file[MAXSTRING], test_predict_file[MAXSTRING];
    
    char parameter_file[MAXSTRING] = "parameters.txt";
    int i = 0;
    if (argc == 1) {
        printf("Drug sensitivity rank prediction \n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-druglist <file>\n");
        printf("\t\tUse <file> to build a dictionary of drug names\n");
        printf("\t-celllist <file>\n");
        printf("\t\tUse <file> to build a dictionary of cell names\n");
        printf("\t-cellfeature <file>\n");
        printf("\t\tUse <file> to get gene features for each cell\n");
        printf("\t-trainpairs <file>\n");
        printf("\t\tUse <file> to get the training pairs\n");
        printf("\t-testpairs <file>\n");
        printf("\t\tUse <file> to get the testing pairs\n");
        printf("\t-testpredict <file>\n");
        printf("\t\tUse <file> to store the predict results\n");
        printf("\nExamples:\n");
        printf("./rank -druglist druglist.txt -celllist celllist.txt -cellfeature cellfeature.txt -trainpairs trainpair.txt -testpairs testpair.txt -testpredict test_predict.csv \n\n");
        return 0;
    }
    
    drug_list_file[0] = 0;
    cell_list_file[0] = 0;
    cell_feature_file[0] = 0;
    train_pairs_file[0] = 0;
    test_pairs_file[0] = 0;
    test_predict_file[0] = 0;
               
    if ((i = ArgPos((char *)"-druglist", argc, argv)) > 0)    strcpy(drug_list_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-celllist", argc, argv)) > 0) 	  strcpy(cell_list_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-cellfeature", argc, argv)) > 0) strcpy(cell_feature_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-trainpairs", argc, argv)) > 0) strcpy(train_pairs_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-testpairs", argc, argv)) > 0) strcpy(test_pairs_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-testpairs", argc, argv)) > 0) strcpy(test_pairs_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-testpredict", argc, argv)) > 0) strcpy(test_predict_file, argv[i + 1]);
               
    ReadDrugDict(drug_list_file);
    ReadCellDict(cell_list_file);
    ReadCellFeature(cell_feature_file);
    ReadTrainPairs(train_pairs_file);
    ReadTestPairs(test_pairs_file);
    InitParameters();
    Train2();
    Test(test_predict_file);
    SaveParameters(parameter_file);
    DestroyParameters();
    return 0;
}