CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2 -Iinclude
LDFLAGS :=

TARGET := main

SRC_DIR := source
INC_DIR := include
BUILD_DIR := build

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/tokenizer_loader.cpp $(SRC_DIR)/tokenizer.cpp $(SRC_DIR)/embedding.cpp $(SRC_DIR)/layernorm.cpp $(SRC_DIR)/linear.cpp $(SRC_DIR)/attention.cpp $(SRC_DIR)/gelu.cpp $(SRC_DIR)/transformer_block.cpp
OBJS := $(BUILD_DIR)/main.o $(BUILD_DIR)/tokenizer_loader.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/embedding.o $(BUILD_DIR)/layernorm.o $(BUILD_DIR)/linear.o $(BUILD_DIR)/attention.o $(BUILD_DIR)/gelu.o $(BUILD_DIR)/transformer_block.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(INC_DIR)/tokenizer.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/main.cpp -o $(BUILD_DIR)/main.o

$(BUILD_DIR)/tokenizer_loader.o: $(SRC_DIR)/tokenizer_loader.cpp $(INC_DIR)/tokenizer.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/tokenizer_loader.cpp -o $(BUILD_DIR)/tokenizer_loader.o

$(BUILD_DIR)/tokenizer.o: $(SRC_DIR)/tokenizer.cpp $(INC_DIR)/tokenizer.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/tokenizer.cpp -o $(BUILD_DIR)/tokenizer.o

$(BUILD_DIR)/embedding.o: $(SRC_DIR)/embedding.cpp $(INC_DIR)/embedding.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/embedding.cpp -o $(BUILD_DIR)/embedding.o

$(BUILD_DIR)/layernorm.o: $(SRC_DIR)/layernorm.cpp $(INC_DIR)/layernorm.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/layernorm.cpp -o $(BUILD_DIR)/layernorm.o

$(BUILD_DIR)/linear.o: $(SRC_DIR)/linear.cpp $(INC_DIR)/linear.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/linear.cpp -o $(BUILD_DIR)/linear.o

$(BUILD_DIR)/attention.o: $(SRC_DIR)/attention.cpp $(INC_DIR)/attention.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/attention.cpp -o $(BUILD_DIR)/attention.o

$(BUILD_DIR)/gelu.o: $(SRC_DIR)/gelu.cpp $(INC_DIR)/gelu.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/gelu.cpp -o $(BUILD_DIR)/gelu.o

$(BUILD_DIR)/transformer_block.o: $(SRC_DIR)/transformer_block.cpp $(INC_DIR)/transformer_block.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/transformer_block.cpp -o $(BUILD_DIR)/transformer_block.o

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

re: clean all

.PHONY: all run clean re