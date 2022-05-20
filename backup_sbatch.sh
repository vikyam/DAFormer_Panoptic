cd /scratch
mkdir cityscapes
mkdir synthia
cd /srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/cityscapes/
echo "starting to expand datasets"
tar -xf gtFine_panoptic.tar -C /scratch/cityscapes
echo "explanded gtFine Cityscapes"
tar -xf leftImg8bit.tar -C /scratch/cityscapes
tar -xf sample_class_stats.json.tar -C /scratch/cityscapes
tar -xf sample_class_stats_dict.json.tar -C /scratch/cityscapes
tar -xf samples_with_class.json.tar -C /scratch/cityscapes
cd /srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/synthia/
tar -xf panoptic-labels-crowdth-0-for-daformer.tar -C /scratch/synthia
tar -xf RGB.tar -C /scratch/synthia
tar -xf sample_class_stats.json.tar -C /scratch/synthia
tar -xf sample_class_stats_dict.json.tar -C /scratch/synthia
tar -xf samples_with_class.json.tar -C /scratch/synthia
cd /scratch/synthia/
ls
echo "starting training"