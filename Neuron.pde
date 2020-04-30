class Neuron {
  float[] weights;
  float bias;
  float value = 0;
  float error;

  Neuron(int weightNum) {
    //Generate random weights and bias
    weights = new float[weightNum];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = random(-1, 1);
    }  
    bias = random(-5, 5);
  }

  void display(int x, int y, int size) {
    strokeWeight(0.5);
    stroke(0);
    fill(255-abs(value)*255, 200);
    ellipse(x, y, size, size);
    fill(0);
    text(value, x, y-10);
  }
}
