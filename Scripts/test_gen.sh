python Scripts/test_gen.py --model /home/zshen5/Models/pascal/test_inference.prototxt --weights /home/zshen5/Models/camvid/test_weights_camvid100k.caffemodel --iter 3352 2>&1 | tee seg_test.log  # Test SegNet
python Scripts/test_resize.py
python Scripts/crf.py /home/zshen5/Data/ImageNet2016/release_test/testing /home/zshen5/Data/ImageNet2016/release_test/predictions_camvid100k/img_ori_size /home/zshen5/Data/ImageNet2016/release_test/predictions_camvid100k/img_ori_size_crf /home/zshen5/Data/ImageNet2016/release_test/predictions_camvid100k/img_ori_size_crf_color 
