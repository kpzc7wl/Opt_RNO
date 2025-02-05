# Opt_RNO


### Dataset
https://drive.google.com/drive/folders/1piGB478DebJ-6u8Pap6kFzc3zJDcPsnr?usp=drive_link

### Training


To train RNO, parameters could be updated using argparser or modifying args.py file

```python
python train_geo.py --gpu 0 --dataset micro2d_sens_100x12 --use-normalizer unit  --normalize_x unit --component all --sobolev \
--loss-name rel2 --epochs 100 --batch-size 8 --model-name RNO \
--optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  \
--grad-clip 1000.0 --n-hidden 128 --n-layers 3  --use-tb 1 \
--seed 2024 --attn-type virtualfourier --n-head 8 --modes 16 --sample-rate 1 --gamma 1 --noref 0.3 --lamb 0.9 --comment _vf_ST0.9
```

Use ``train_geo.py`` for Microreator2D, ``train_shape.py`` for Fuelcell2D and Inductor2D, ``train_3d.py`` for Drone3D. 

Use ``--sobolev`` to enable Sobolev training with sensitivity data.  

Use``--attn-type virtualfourier`` to enable virtualfourier layer.  



### Test Samples
![Microreactor](/figure/micro_gt.svg)
![Microreactor_vf](/figure/micro_vf.svg)
![Microreactor_vf_noref](/figure/micro_vf_noref.svg)

Test sample from Microreactor2D. The inlet and outlet are cropped in order to show the region of reaction more clearly. First row is the ground truth. Second row is the prediction from R-VF. The last row is the prediction from R-VF without reference. The columns from the left are density $\theta$, pressure $p$, velocity components $u$, $v$, concentration $c$, and sensitivity.

![Fuelcell](/figure/fuelcell_gt.svg)
![Fuelcell_vf](/figure/fuelcell_vf.svg)
![Fuelcell_vf_noref](/figure/fuelcell_vf_noref.svg)

Test sample from Fuelcell2D. First row is the ground truth. Second row is the prediction from R-VF. The last row is the prediction from R-VF without reference. The columns from the left are the mask of internal walls, pressure $p$, velocity components $u$, $v$, and mesh shift $dx$ and $dy$.

![Inductor](/figure/inductor_gt.svg)
![Inductor_vf](/figure/inductor_vf.svg)
![Inductor_vf_noref](/figure/inductor_vf_noref.svg)

Test sample from Inductor2D. First row is the ground truth. Second row is the prediction from R-VF. The last row is the prediction from R-VF without reference. The columns from the left are the mask of magnetic core (the mesh of coils are 0's), $B_r$, $B_z$,  and mesh shift $dr$ and $dz$.