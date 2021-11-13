FROM  pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "/bin/bash"]
CMD [ "cd",'src']
