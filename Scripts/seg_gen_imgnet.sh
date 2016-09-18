python Scripts/seg_gen_imgnet.py --model /home/zshen5/Models/pascal/inference.prototxt --weights /home/zshen5/Models/camvid/test_weights_camvid100k.caffemodel --iter 2000 2>&1 | tee seg.log  # Test SegNet
python Scripts/resize.py
python Scripts/crf.py /home/zshen5/Data/ImageNet2016/ADEChallengeData2016/images/validation /home/zshen5/Data/ImageNet2016/ADEChallengeData2016/predictions_camvid100k/img_ori_size /home/zshen5/Data/ImageNet2016/ADEChallengeData2016/predictions_camvid100k/img_ori_size_crf /home/zshen5/Data/ImageNet2016/ADEChallengeData2016/predictions_camvid100k/img_ori_size_crf_rgb 
