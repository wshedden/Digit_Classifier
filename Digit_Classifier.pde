//784-500-10 epochs:2 batch_size:5000 learning_rate:0.01 accuracy:91%
//784-500-10 epochs:2 batch_size:5000 learning_rate:0.1 accuracy:86%
//784-500-10 epochs:2 batch_size:5000 learning_rate:0.001 accuracy:62%
Network net;

String instructions = "Left click on the canvas to draw; right click to reset.\nPress space to toggle drawing mode (HIGHLY RECOMMENDED).";

final int[] dimensions = new int[]{784, 500, 10};
float[][] expected;
float[][] inputs;
String[] data;
int[] digits;

int[][] grid = new int[28][28];

int prediction;
float xStart = 600;
float yStart = 100;
final float tileSize = 20;


boolean display = true;

void setup() {
  size(1200, 800);
  background(255);
  
  for (int i = 0; i < grid.length; i++) {
    for (int k = 0; k < grid[i].length; k++) {
      grid[i][k] = 0;
    }
  }
  
  net = new Network(dimensions);
  net.loadParameters("a94.txt");
  //Training
  //int epochs = 5;
  //int batchSize = 50;
  //float learningRate = 0.0001;
  //int randomStart = (int) random(0, 50000);
  
  //for(int i = 0; i < 200; i++){
  //  train(net, batchSize, epochs, learningRate, batchSize*i+randomStart, false);
  //  float accuracy = test(net, 100, true);
  //  println(accuracy, learningRate);
    
  //  net.saveParameters(String.format("Sigmoid_tests/a=%f.txt", accuracy));
  //}
  //loadData(loadStrings("mnist_test.csv"));
  loadData(loadStrings("example_digit.txt"));
  grid = inputToGrid(inputs[0]);
}

void draw() {
  background(160, 150, 165);
  updateGrid(grid, tileSize, xStart, yStart);
  

  strokeWeight(5);
  stroke(156, 19, 168);
  rect(xStart-1, yStart-1, tileSize*28+2, tileSize*28+2);
  displayGrid(grid, 20, xStart, yStart);


  if(display){
    xStart=width/2;
    net.setInputs(gridToInput(grid));
    net.propagate();
    prediction = arrayToDigit(net.getOutputs());
    net.display();
    fill(0);
    
  } else{
     xStart=width/2-28*tileSize/2; 
  }
}

void keyPressed() {
  if (key == ' ') {
    display = !display;
  }
}
