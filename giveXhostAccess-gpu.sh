export containerId=`sudo nvidia-docker ps -a -q -l --filter ancestor=palebone/tf-opencv-docker-gpu:version01`
sudo xhost +local:`sudo docker inspect --format='{{ .Config.Hostname }}' $containerId`
sudo nvidia-docker start $containerId 
sudo nvidia-docker attach $containerId
