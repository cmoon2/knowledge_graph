# Knowledge Graph Completion

This repository holds the code and datasets for experiments in the two papers (Please, cite these publication if you use our source codes and datasets):

[1 - ContE Model] Changsung Moon, Steve Harenberg, John Slankas, and Nagiza F. Samatova. ”Learning Contextual Embeddings for Knowledge Graph Completion." In the Pacific Asia Conference on Information Systems (PACIS). 2017. 

[2 - ETE Model] Changsung Moon, Paul Jones, and Nagiza F. Samatova. ” Learning Entity Type Embeddings for Knowledge Graph Completion." In the ACM International Conference on Information and Knowledge Management (CIKM). 2017.


There are three kinds of inference of missing data in a knowledge graph: 1) missing entity, 2) missing relation type and 3) missing entity type


Author: Changsung Moon (cmoon2@ncsu.edu)


Dependency Requirements: Python==2.7


The source codes are modified versions from original source code from https://github.com/mnick/holographic-embeddings



Usage instructions:

1. Parameter Setup
	
* 1.1 Related files

		1.1.1 ./kg/run_conte.py (for our ContE model)
		1.1.2 ./kg/run_ete.py (for our ETE model)
		1.1.3 ./kg/run_transe.py (for the baseline TransE)
		1.1.4 ./kg/run_rescal.py (for the baseline Rescal)
		1.1.5 ./kg/run_hole.py (for the baseline HolE)
		1.1.6 ./kg/run_null.py (for the baseline Null model)
		1.1.7 If you see an error that is related to "import", 
		      uncomment followings and put the right path of your folder of source codes in the above files:
			
			#import sys
			#sys.path.append("[PATH OF THE FOLDER OF SOURCE CODES]")
			#print(sys.path)

* 1.2 Parameter default values for ContE and ETE models

		1.2.1 --ncomp = 200 (Size of embedding vector)
		1.2.2 --margin = 2.0 (Margin for loss function)
		1.2.3 --lr = 0.1 (Learning rate)
		1.2.4 --me (Maximum number of epochs)
		1.2.5 --ne = 1 (Number of negative examples)
		1.2.6 --nb = 100 (Number of batches)
		1.2.7 --fout (Path to store model and results)
		1.2.8 --fin (Path to input a trained model such as ContE model)
		1.2.9 --kg_train (Path to input training data)
		1.2.10 --kg_valid (Path to input validation data)
		1.2.11 --kg_test (Path to input test data)
		1.2.12 --test-all = 100 (Evaluate test set after x epochs)
		1.2.13 --mode 
			= 'rank' for Inference of missing entity in run_conte.py
			= 'p_rank' for Inference of missing relation type in run_conte.py
			= 'et_rank' for Inference of missing entity type in run_ete.py
		1.2.14 --sampler = 'random-mode' (Data sampling)

2. Inference of missing entity (ContE model)

* 2.1 Related datasets

		2.1.1 Freebase: ../datasets/Freebase/FB15k/
			2.1.1.1 FB15k is published by "Translating Embeddings for Modeling Multi-relational Data (2013)."
		2.1.2 YAGO: ../datasets/YAGO/YAGO43k/

* 2.2 How to run

		2.2.1 Set up parameters especially
			--me, --fout, --kg_train, --kg_valid, --kg_test, --test-all, --mode='rank'
		2.2.2 Train the ContE model
			python ./kg/run_conte.py

3. Inference of missing relation type (ContE model)

* 3.1 Related datasets

		3.1.1 Freebase: ../datasets/Freebase/FB15k/
		3.1.2 YAGO: ../datasets/YAGO/YAGO43k/

* 3.2 How to run

		3.2.1 Set up parameters especially
			--me, --fout, --kg_train, --kg_valid, --kg_test, --test-all, --mode='p_rank'
		3.2.2 Train the ContE model
			python ./kg/run_conte.py

4. Inference of missing entity type (ETE model)

* 4.1 Related datasets

		4.1.1 Freebase: ../datasets/Freebase/FB15k/
		4.1.2 YAGO: ../datasets/YAGO/YAGO43k/
		4.1.3 ../datasets/Freebase/Missing_Entity_Types/
		4.1.4 ../datasets/YAGO/Missing_Entity_Types/
	
* 4.2 How to run

		4.2.1 Train the ContE model with 4.1.1 or 4.1.2 dataset
			python ./kg/run_conte.py
		4.2.2 Use the output model of 4.2.1 as the input of the ETE model
			4.2.2.1 Set up the parameters in the file "./kg/run_ete.py"
				--me, --fout, --fin, --kg_train, --kg_valid, --kg_test, --test-all, --mode='et_rank'
				"--fin" as the path of the output model 
			4.2.2.2 Tran the ETE model
				python ./kg/run_ete.py


