sudo nvidia-docker run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --device="/dev/video0:/dev/video0" \
    --volume="/home/iansp/dockerFiles/sharedDockerFiles:/root/sharedDockerFiles" \
    --volume="/etc/machine-id:/etc/machine-id" \
    palebone/tf-opencv-docker-gpu:version01 \
    bash 
sudo export containerId=$(docker ps -l -q)
sudo xhost +local:`sudo nvidia-docker inspect --format='{{ .Config.Hostname }}' $containerId`
sudo nvidia-docker start $containerId
