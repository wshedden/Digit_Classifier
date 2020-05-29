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
    textSize(14);
    text(instructions, 275, 40);
    net.setInputs(gridToInput(grid));
    net.propagate();
    prediction = arrayToDigit(net.getOutputs());
    net.display();
    fill(255);
    textSize(100);
    fill(0);
    stroke(156, 19, 168);
    strokeWeight(2);
    rect(width/2-170, height/2-88, 100, 100);
    fill(255);
    text(prediction, width/2-150, height/2);
    
  } else{
     xStart=width/2-28*tileSize/2; 
  }
}

void keyPressed() {
  if (key == ' ') {
    display = !display;
  }
}
