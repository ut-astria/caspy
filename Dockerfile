from orbdetpy:2.1.0

ENV HOME=/root

COPY caspy ${HOME}/caspy/caspy
COPY utils ${HOME}/caspy/utils
COPY LICENSE ${HOME}/caspy/LICENSE
COPY requirements.txt ${HOME}/caspy/requirements.txt

RUN cd && \
    . env_orbdetpy/bin/activate && \
    cd ${HOME}/caspy && \
    pip install -r requirements.txt

ENTRYPOINT cd && \
    . env_orbdetpy/bin/activate && \
    cd ${HOME}/caspy/caspy && \
    python cas.py

