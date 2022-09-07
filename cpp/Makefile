CXX=g++
#SOURCE=main.cpp sparse/sparse_layer.cpp sparse/sparse_model.cpp sparse/layer.cpp sparse/dense_layer.cpp autoencoder/compressed_batch.cpp autoencoder/autoencoder.cpp
INCLUDES := -I${EIGEN_DIR} -I${JSON_DIR} -I./ #-Isparse/ -Inormalization/ -Iautoencoder/ -Iutils/
BUILD_DIR := ./build
OBJS := $(BUILD_DIR)/sparse_layer.o $(BUILD_DIR)/sparse_layer.o $(BUILD_DIR)/dense_layer.o  $(BUILD_DIR)/layer.o $(BUILD_DIR)/sparse_model.o $(BUILD_DIR)/compressed_batch.o $(BUILD_DIR)/autoencoder.o  

all: $(BUILD_DIR)/libautoencoder_c.so test

$(BUILD_DIR)/sparse_layer.o: sparse/sparse_layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -lpthread -march=native -O3 

$(BUILD_DIR)/sparse_model.o: sparse/sparse_model.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -lpthread -march=native -O3 

$(BUILD_DIR)/layer.o: sparse/layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -lpthread -march=native -O3

$(BUILD_DIR)/dense_layer.o: sparse/dense_layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -lpthread -march=native -O3

$(BUILD_DIR)/compressed_batch.o: autoencoder/compressed_batch.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -lpthread -march=native -O3

$(BUILD_DIR)/autoencoder.o: autoencoder/autoencoder.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -lpthread -march=native -O3

$(BUILD_DIR)/autoencoder_interface_c.o: autoencoder/autoencoder_interface_c.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -lpthread -march=native -O3

$(BUILD_DIR)/libautoencoder_c.so: $(BUILD_DIR)/autoencoder_interface_c.o $(BUILD_DIR)/autoencoder.o
	$(CXX) ${INCLUDES} -shared -o $@ -lpthread $^ -march=native -O3

test: main.cpp $(OBJS)
	$(CXX) ${INCLUDES} $^ -o $@ -march=native -O3
