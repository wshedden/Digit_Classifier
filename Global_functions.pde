float sigmoid(float x, boolean prime) {
  float f = (float)(1 / (1 + Math.exp(-x)));
  if (prime) return f * (1-f);
  return f;
}

float sigmoid_inv(float x){
    return -log(1/x-1);
}

float relu(float x, boolean prime){
  if(x < 0){
    if(prime) return 0.01;
    return x*0.01;
  } else if(prime) return 1;
  return x;
}

float fastSigmoid(float x, boolean prime) {
  float f = 0.5 * (x  / (1 + abs(x))) + 0.5;
  if (prime) return f * (1-f);
  return f;
}

float crossMultiply(float[] a, float[] b) {
  float sum = 0;
  for (int i = 0; i < a.length; i++) {
    sum += a[i]*b[i];
  }
  return sum;
}

float sum(float[] arr) {
  float total = 0;
  for (int i = 0; i < arr.length; i++) {
    total += abs(arr[i]); //NOTE: Absolute value is just in case there are negative weights
  }
  return total;
}

float[] digitToArray(int n) {
  float[] arr = new float[10];
  for (int i = 0; i < arr.length; i++) {
    arr[i] = 0;
  }
  arr[n] = 1;
  return arr;
}

int arrayToDigit(float[] arr) {
  float max = 0;
  int maxDigit = -1;
  for (int i = 0; i < arr.length; i++) {
    if (arr[i] > max) {
      max = arr[i];
      maxDigit = i;
    }
  }
  return maxDigit;
}


void loadData(String[] file) {
  inputs = new float[file.length][784];
  digits = new int[file.length];
  expected = new float[file.length][10];
  for (int i = 0; i < file.length; i++) {
    String[] line = split(file[i], ',');
    for (int k = 1; k < 784; k++) {
      inputs[i][k] = Float.parseFloat(line[k]);
    }
    expected[i] = digitToArray(Integer.parseInt(line[0]));
  }
}

void train(Network net, int batchSize, int epochs, float learning_rate, int offset, boolean print) {
  loadData(loadStrings("mnist_train.csv"));
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int input = 0; input < batchSize; input++) {
      if (print)
        println(String.format("Epoch %d/%d: %f percent complete", epoch+1, epochs, 100*(float)(input+1)/(float)batchSize));
      net.setInputs(inputs[input+offset]);
      net.propagate();
      net.backpropagate(expected[input+offset], learning_rate);
    }
  }
}

float test(Network net, int testNum, boolean print) {
  loadData(loadStrings("mnist_test.csv"));
  float[][] digits = new float[10][3];
  int correct = 0;
  for (int input = 0; input < testNum; input++) {
    if (print)
      println(String.format("Test %d/%d", input+1, testNum));
    net.setInputs(inputs[input]);
    net.propagate();
    int digit = arrayToDigit(net.getOutputs());
    int expectedDigit = arrayToDigit(expected[input]);
    if (digit == expectedDigit) {
      correct++;
      digits[expectedDigit][0]++;
    } else {
      digits[expectedDigit][1]++;
      println(String.format("Confused %d with %d", expectedDigit, digit));
    }
  }
  float[] accuracies = new float[digits.length];
  for (int i = 0; i < digits.length; i++) {
    digits[i][2] = digits[i][0]/(digits[i][0] + digits[i][1]);
    accuracies[i] = digits[i][2];
    println(String.format("%d: %f", i, digits[i][2]));
  }  
  println(String.format("Accuracy score = %f", sum(accuracies)));

  return (float) correct / (float)testNum;
}

float[] gridToInput(int[][] arr) {
  float[] input = new float[784];
  for (int i = 0; i < arr.length; i++) {
    for (int k = 0; k < arr[i].length; k++) {
      input[i*28+k] = arr[k][i];
    }
  }
  return input;
}

int[][] inputToGrid(float[] arr) {
  int[][] grid = new int[28][28];
  for (int i = 0; i < arr.length; i++) {
    grid[i%28][i/28] = (int) arr[i];
  }
  return grid;
}

void displayGrid(int[][] grid, float tileSize, float xStart, float yStart) {
  for (int i = 0; i < grid.length; i++) {
    for (int k = 0; k < grid[i].length; k++) {
      strokeWeight(0.2);
      fill(grid[i][k]);
      stroke(grid[i][k]);
      rect(i*tileSize+xStart, k*tileSize+yStart, tileSize, tileSize);
    }
  }
}

void updateGrid(int[][] grid, float tileSize, float xStart, float yStart) {
  int[][] offsets = new int[][]{new int[]{0, 1}, new int[]{1, 0}, new int[]{-1, 0}, new int[]{0, -1}};
  for (int i = 0; i < grid.length; i++) {
    for (int k = 0; k < grid[i].length; k++) {
      if (mouseX >= i*tileSize+xStart && mouseX <= i*tileSize+tileSize+xStart && mousePressed) {
        if (mouseY >= k*tileSize+yStart && mouseY <= k*tileSize+tileSize+yStart && mouseButton == LEFT) {
          grid[i][k] = 255;
          for (int offset = 0; offset < offsets.length; offset++) {
            try {
              int value = grid[i+offsets[offset][0]][k+offsets[offset][1]];
              if(value < 180){
                value = 180;
              } else if(value == 180) value = 240;
              grid[i+offsets[offset][0]][k+offsets[offset][1]] = value;
            }
            catch(Exception e) {
            }
          }
        }
      }
      if (mousePressed && mouseButton == RIGHT) {
        grid[i][k] = 0;
      }
    }
  }
}
