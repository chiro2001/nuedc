# nuedc

[toc]

## 采购清单

1. 最终结果

   1. 千兆交换机：支持/不支持POE

      1. 千兆网线（x3）

   2. 两个网口摄像机

      1. 高帧率？应当至少60fps
      2. 高分辨率？1080P
      3. 选型：（还没选好）

         1. https://item.taobao.com/item.htm?spm=a230r.1.14.57.24c04122yZLLe6&id=638493054601&ns=1&abbucket=12#detail
         2. ![img](README.assets/8LDO48C$8@[GWU0353$FOVS.png)https://item.taobao.com/item.htm?id=599528900176&ali_refid=a3_430582_1006:1240250154:N:vAnE48VDVehafmeTVx9zCR8jCZxd3bvD:9b600939f10681ce55998c81c3bc1ae3&ali_trackid=1_9b600939f10681ce55998c81c3bc1ae3&spm=a230r.1.14.6#detail
         3. ![img](README.assets/8LDO48C$8@[GWU0353$FOVS-1635990902959.png)https://item.taobao.com/item.htm?spm=a230r.1.14.18.24c04122yZLLe6&id=594194622824&ns=1&abbucket=12#detail
      4. 选型：选好的（基本是两家店铺选一家买，同款）
         1. 相机：MV-CA013-20GC
            1. https://item.taobao.com/item.htm?spm=a1z10.1-c-s.w4004-21918332539.20.19b27b91FDYdIj&id=599529224721
            2. https://item.taobao.com/item.htm?spm=a230r.1.14.16.279858e9rAAoq5&id=558631943461&ns=1&abbucket=12#detail
         2. 镜头：MVL-MF0828M-8MP
            1. https://item.taobao.com/item.htm?spm=a230r.1.14.22.19524fc8y7aByG&id=656223074552&ns=1&abbucket=12#detail
            2. https://item.taobao.com/item.htm?spm=a1z10.1-c-s.w4004-21918332539.20.19b27b91FDYdIj&id=599529224721
      5. 镜头参数

         1. 焦距计算：$f=\frac{h\times WD}{H}$

         2. MV-CE013-50GM

            1. 靶面尺寸：1/3"
            2. 宽4.8mm*高3.6mm，对角线6mm
            3. $\frac{3.6mm\times 1m}{1m}=3.6mm$
            4. 选择焦距小的镜头：8mm
            5. https://item.taobao.com/item.htm?spm=a1z10.1-c-s.w4004-21918332539.20.19b27b91FDYdIj&id=599529224721

   3. 激光笔（<10cm）

   4. 激光切割的支架（木板，最大能切多长）

2. 已经有的

   1. 树莓派4B
   2. 用手机的摄像头进行测试
   3. 用笔记本先做主机的测试

## 问题

1. 能不能用树莓派？能。

## 器件数据/参数

**视觉部分：**

![img](https://img.alicdn.com/imgextra/i4/879125025/O1CN014gBQYv1mzUqHJN5rN_!!879125025.png)

![img](https://img.alicdn.com/imgextra/i2/879125025/O1CN01gS6ZUf1mzUq6vzQ4z_!!879125025.png)



![img](https://img.alicdn.com/imgextra/i1/2200639829572/O1CN01gEBDsz2Ka1mte8j0Z_!!2200639829572.jpg)

![img](https://img.alicdn.com/imgextra/i3/2200639829572/O1CN01zGlpF22Ka1n5TUuDc_!!2200639829572.jpg)



## 可行性测试

1. 相机矫正
2. 网口相机调试
3. 框选运动的激光笔
4. 找到移动的激光笔发出的激光的地面位置
5. 两个相机得到两条射线之后求出交点
6. 算法物理跟踪

## 进度规划

