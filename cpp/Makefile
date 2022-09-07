CXX=g++
SOURCE=main.cpp sparse/sparse_layer.cpp sparse/sparse_model.cpp sparse/layer.cpp sparse/dense_layer.cpp autoencoder/compressed_batch.cpp autoencoder/autoencoder.cpp
INCLUDE=-I${EIGEN_DIR} -I${JSON_DIR} -Isparse/ -Inormalization/ -Iautoencoder/ -Iutils/
OUTPUT=test

default:
	${CXX} ${SOURCE} ${INCLUDE} -o ${OUTPUT} -march=native
