# sinlge function.
generate_segmentation:
	python generate_segmentation.py
verify_segmentation:
	python verify_segmentation.py
predictiton:
	rm -rf history.log
	python prediction.py
evaluation:
	python evaluation.py

# batchmark
test_baseline:
	rm record
	rm -rf history.log
	python test_baseline.py
test_prediciton:
	rm record
	python test_prediction.py

# remote access
ssh_to_server:
	ssh tlin@lsir-cluster-08.epfl.ch
cp_all_to_lsir_server:
	rsync -av -e ssh --exclude='data/output/training/*' --exclude='archive*' ../code tlin@lsir-cluster-08.epfl.ch:~/
cp_all_to_g5k_server:
	rsync -av -e ssh --exclude='data/output/training/*' --exclude='archive*' ../code tlin@access.grid5000.fr:rennes/test/
cp_all_to_gcloud:
	gcloud compute copy-files ../code hpc-highmem-16:
