#BASE image
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.7 python3-pip python3.7

#Setting up the work directory
WORKDIR /test
ARG var
# ENV var=${var}

#copy local requirements.txt to docker repo inside test/ folder 
COPY requirements.txt /test/requirements.txt
RUN pip install -r /test/requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update
RUN apt-get install vim -y

#now we will copy arg variable in our work directory
COPY ${var} /test/${var}
#copy the shell script
COPY run.sh /test/run.sh 
COPY hyperparameters.ini /test/hyperparameters.ini

#copy the aug script
COPY image_augs /test/image_augs
#copy the mainfile
COPY mainfile.py /test/mainfile.py
# CMD [ "bash","/test/run.sh" ]



####docker command for build
#docker build -t <name>:<tag_name> .   (here . means same directory)

###docker command for run
#docker run --rm -e configFile=hyperparameters.ini app:3


#del all images --> docker rm -vf $(docker ps -aq)


####for common issues

#socket problem --> sudo chmod 666 /var/run/docker.sock
#https://github.com/palantir/gradle-docker/issues/188
#https://stackoverflow.com/questions/68673221/warning-running-pip-as-the-root-user
#https://stackoverflow.com/questions/65895928/how-to-delete-a-docker-image
