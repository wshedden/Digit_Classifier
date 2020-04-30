class Network {
  Neuron[][] neurons;
  int[] dimensions;
  int totalNeurons = 0;
  Network(int[] dimensions) {
    //Declaring neurons
    this.dimensions = dimensions;
    neurons = new Neuron[dimensions.length][];
    for (int layer = 0; layer < dimensions.length; layer++) {
      neurons[layer] = new Neuron[dimensions[layer]];
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
        int weightNum = 0;
        if (layer > 0) {
          weightNum = neurons[layer-1].length;
        }
        neurons[layer][neuron] = new Neuron(weightNum);
        totalNeurons++;
      }
    }
  }

  void propagate() {
    for (int layer = 1; layer < neurons.length; layer++) {
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
        Neuron currentNeuron = neurons[layer][neuron];
        Neuron[] prevLayer = neurons[layer-1];
        //Get the values for the previous layer
        float[] prevLayerValues = new float[prevLayer.length];
        for (int i = 0; i < prevLayer.length; i++) {
          prevLayerValues[i] = prevLayer[i].value;
        }
        neurons[layer][neuron].value = sigmoid(crossMultiply(currentNeuron.weights, prevLayerValues) + neurons[layer][neuron].bias, false);
      }
    }
    //normalise(); 
  }


  void display() {
    textSize(12);
    for (int layer = 0; layer < neurons.length; layer++) {
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
        int x = (int) layer*80+40;
        int y = (int) neuron*40+height/(4*neurons[layer].length)+30;
        if (x < width && y < height-50)
          neurons[layer][neuron].display(x, y, 20);
        else {
          ellipse(x, y-5, 1.5, 1.5);
          ellipse(x, y+8, 1.5, 1.5);
          ellipse(x, y+19, 1.5, 1.5);
        }
        //Display lines
        if (layer > 0) {
          strokeWeight(0.5);
          stroke(0);
          Neuron[] prevNeurons = neurons[layer-1];
          for (int i = 0; i < prevNeurons.length; i++) {
            int x2 = (layer-1)*80+40;
            int y2 = i*40+height/(4*neurons[layer-1].length);
            if (x < width && y < height-50 && x2 < width && y2 < height-50 && y > 5 && y2 > 5)
              line(x, y, x2, y2);
          }
        }
      }
    }
  }

  void setInputs(float[] inputValues) {
    for (int neuron = 0; neuron < neurons[0].length; neuron++) {
      neurons[0][neuron].value = inputValues[neuron];
    }
  }

  void backpropagate(float[] expected, float learningRate) {
    calculateError(expected, false);
    int n = neurons.length;
    for (int layer = n-1; layer > 0; layer--) {
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {

        neurons[layer][neuron].bias += learningRate * neurons[layer][neuron].error * sigmoid(neurons[layer][neuron].value, true);

        for (int weight = 0; weight < neurons[layer][neuron].weights.length; weight++) {
          float delta;

          delta = learningRate * neurons[layer][neuron].error * neurons[layer-1][weight].value * sigmoid(neurons[layer][neuron].value, true);

          neurons[layer][neuron].weights[weight] += delta;
        }
      }
    }
  }

  void calculateError(float[] expected, boolean square) {
    //TODO: There is a possibility that weightSum should actually be the ABSOLUTE sum of the weights
    int n = neurons.length;
    for (int layer = 0; layer < neurons[n-1].length; layer++) {
      if (square)
        neurons[n-1][layer].error = sq(expected[layer]-neurons[n-1][layer].value);
      else neurons[n-1][layer].error = expected[layer]-neurons[n-1][layer].value;
    }
    for (int layer = n-2; layer >= 0; layer--) {
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
        float error = 0;
        for (int neuron_out = 0; neuron_out < neurons[layer+1].length; neuron_out++) {
          float weightSum = sum(neurons[layer+1][neuron_out].weights);
          float weightProportion = neurons[layer+1][neuron_out].weights[neuron]/weightSum;
          error += neurons[layer+1][neuron_out].error * weightProportion;
        }
        neurons[layer][neuron].error = error;
      }
    }
  }

  float[] getOutputs() {
    int n = neurons.length;
    float[] outputs = new float[neurons[n-1].length];
    for (int neuron = 0; neuron < neurons[n-1].length; neuron++) {
      outputs[neuron] = neurons[n-1][neuron].value;
    }
    return outputs;
  }

  void loadParameters(String file) {
    String[] lines = loadStrings(file);
    float[][] values = new float[totalNeurons][];
    for (int i = 0; i < lines.length; i++) {
      String[] line = split(lines[i], ',');
      values[i] = new float[line.length];
      for (int k = 0; k < line.length; k++) {
        values[i][k] = Float.parseFloat(line[k]);
      }
    }
    for (int layer = 1; layer < neurons.length; layer++) {
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
        neurons[layer][neuron].bias = values[(layer-1)*dimensions[layer-1]+neuron][0];
        for (int prevNeuron = 0; prevNeuron < neurons[layer-1].length; prevNeuron++) {
          neurons[layer][neuron].weights[prevNeuron] = values[(layer-1)*dimensions[layer-1]+neuron][prevNeuron+1];
        }
      }
    }
  }

  void saveParameters(String file) {
    PrintWriter writer = createWriter(file);
    String[][] lines = new String[neurons.length-1][];
    for (int layer = 1; layer < neurons.length; layer++) {
      lines[layer-1] = new String[neurons[layer].length];
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
        StringBuilder sb = new StringBuilder(Float.toString(neurons[layer][neuron].bias)+',');
        for (int weight = 0; weight < neurons[layer][neuron].weights.length; weight++) {
          sb.append(Float.toString(neurons[layer][neuron].weights[weight]));
          if (weight < neurons[layer][neuron].weights.length-1)
            sb.append(',');
        }
        lines[layer-1][neuron] = sb.toString();
      }
    }
    for (int i = 0; i < lines.length; i++) {
      for (int k = 0; k < lines[i].length; k++) {
        writer.println(lines[i][k]);
      }
    }
    writer.flush();
    writer.close();
  }
  
  void normalise(){
    float[] outputs = getOutputs();
    float sum = sum(outputs);
    for(int i = 0; i < outputs.length; i++){
       neurons[neurons.length-1][i].value = outputs[i]/sum; 
    }  
  }
}
