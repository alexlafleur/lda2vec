lda2vec: Tools for interpreting natural language
=================================================

Dockerfile with prerequsites:
_____________________________

::

  FROM nvidia/cuda

  RUN	apt update && apt-get -y upgrade
  RUN 	apt-get -y install python-pip
  RUN	apt-get -y install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev
  RUN	pip install -U setuptools
  RUN	pip install numpy
  RUN	pip install scipy
  RUN	pip install pandas
  RUN	pip install pyxdameraulevenshtein
  RUN	pip install sklearn
  RUN	pip install spacy
  RUN	pip install h5py
  RUN	pip install cupy --no-cache-dir -vvvv
  RUN	pip install gensim
  RUN	pip install chainer --no-cache-dir -vvvv


ENTRYPOINT /bin/bash


To Do
____________________________

- try lda2vec with german Word2Vec Vector:

  - pre-built: https://tubcloud.tu-berlin.de/public.php?service=files&t=dc4f9d207bcaf4d4fae99ab3fbb1af16
  - build our own using https://github.com/devmount/GermanWordEmbeddings
  - How to handle Emoticons? See here:
      - https://apps.timwhitlock.info/emoji/tables/unicode#block-1-emoticons
      - https://github.com/laurenancona/twimoji/blob/gh-pages/twitterEmojiProject/emoticon_conversion_noGraphic.csv CSV File containing Unicode Emoticons with Description
