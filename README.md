板端需要安装opencv

cd build 
rm -rf *
cmake ..
make   //生成执行文件track
mv track
./track  // 可带参数，可不带

main.cpp中 注释这行，表示读取摄像头的图像，取消注释，匹配单张图片
// #define IMAGE_DEMO
