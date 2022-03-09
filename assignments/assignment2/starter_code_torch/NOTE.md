

docker run --rm -it -v `pwd`:/opt/work/ -e DISPLAY=${ip}:0 -v /tmp/.X11-unix:/tmp/.X11-unix mebusy/gym q2_schedule.py
