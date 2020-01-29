FROM continuumio/miniconda3

RUN conda create -n env python=3.6
RUN activate env
ENV PATH /opt/conda/envs/env/bin:$PATH
	
COPY . ./ADDMo

RUN pip install -e ./ADDMo

EXPOSE 8081
   
CMD [ "python", "-u", "./ADDMo/02_Tool/GUI_Remi.py" , "-p 8081"]

