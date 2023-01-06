CXX=icpc
#SOURCE=main.cpp sparse/sparse_layer.cpp sparse/sparse_model.cpp sparse/layer.cpp sparse/dense_layer.cpp autoencoder/compressed_batch.cpp autoencoder/autoencoder.cpp
INCLUDES := -I${EIGEN_DIR} -I${JSON_DIR} -I./ #-Isparse/ -Inormalization/ -Iautoencoder/ -Iutils/
BUILD_DIR := ./build
OBJS := $(BUILD_DIR)/sparse_layer.o $(BUILD_DIR)/sparse_layer.o $(BUILD_DIR)/dense_layer.o  $(BUILD_DIR)/layer.o $(BUILD_DIR)/sparse_model.o $(BUILD_DIR)/compressed_batch.o $(BUILD_DIR)/autoencoder.o $(BUILD_DIR)/compression_base.o $(BUILD_DIR)/batch_preparer.o  $(BUILD_DIR)/autoencoder_interface_c.o  ${BUILD_DIR}/autoencoder_debug.o

all: $(BUILD_DIR)/libautoencoder_c.so test

$(BUILD_DIR)/sparse_layer.o: sparse/sparse_layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -march=native -O1 

$(BUILD_DIR)/sparse_model.o: sparse/sparse_model.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -march=native -O1 

$(BUILD_DIR)/layer.o: sparse/layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -march=native -O1

$(BUILD_DIR)/dense_layer.o: sparse/dense_layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native -O1

$(BUILD_DIR)/compressed_batch.o: batch_preparation/compressed_batch.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native -O1

$(BUILD_DIR)/batch_preparer.o: batch_preparation/batch_preparer.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native -O1 

$(BUILD_DIR)/compression_base.o: autoencoder/compression_base.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native -O1 

$(BUILD_DIR)/autoencoder.o: autoencoder/autoencoder.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native -O1 

$(BUILD_DIR)/autoencoder_debug.o: autoencoder/autoencoder_debug.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native -O1 

$(BUILD_DIR)/autoencoder_interface_c.o: autoencoder/autoencoder_interface_c.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native -O1 

$(BUILD_DIR)/libautoencoder_c.so: $(OBJS)
	$(CXX) ${INCLUDES}  -shared -o $@ $^ -march=native -O1 -lpthread 

test: main.cpp $(BUILD_DIR)/libautoencoder_c.so
	$(CXX) ${INCLUDES} $^ -o $@ -march=native -O1 -L./build -lautoencoder_c
