CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2 -Iinclude
LDFLAGS :=

TARGET := main

SRC_DIR := source
INC_DIR := include
BUILD_DIR := build

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/tokenizer_loader.cpp $(SRC_DIR)/tokenizer.cpp $(SRC_DIR)/embedding.cpp $(SRC_DIR)/layernorm.cpp
OBJS := $(BUILD_DIR)/main.o $(BUILD_DIR)/tokenizer_loader.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/embedding.o $(BUILD_DIR)/layernorm.o

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

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

re: clean all

.PHONY: all run clean re