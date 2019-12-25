float[][] w1;
float[][] b1;
float[][] w2;
float[][] b2;
float[][] w3;
float[][] b3;

PGraphics canvas;
PGraphics visualization;

float[][] inputMat = new float[1][784];
boolean visualizeLines = false;
import peasy.PeasyCam;
PeasyCam visualizationCam;

void setup() {
  size(1000, 500, P3D);
  canvas = createGraphics(28, 28);
  visualization = createGraphics(500, 500, P3D);
  canvas.beginDraw();
  canvas.background(0);
  canvas.endDraw();
  visualization.beginDraw();
  visualization.background(0);
  visualization.endDraw();

  visualizationCam = new PeasyCam(this, visualization, 400);

  w1 = loadMat("weight1.txt");
  w2 = loadMat("weight2.txt");
  w3 = loadMat("weight3.txt");
  b1 = loadMat("biases1.txt");
  b2 = loadMat("biases2.txt");
  b3 = loadMat("biases3.txt");
}

void draw() {
  canvas.beginDraw();
  if (mousePressed) {
    canvas.stroke(255);
    canvas.strokeWeight(1.8);
    //draw line on the canvas
    canvas.line(
      28*((float)(mouseX-visualization.width)/visualization.width), 
      28*((float)mouseY/visualization.height), 
      28*((float)(pmouseX-visualization.width)/visualization.width), 
      28*((float)pmouseY/visualization.height)
      );
  }
  canvas.loadPixels();

  //canvas to 1d inputMat
  for (int i = 0; i < 28; i ++) {
    for (int j = 0; j < 28; j ++) {
      int row = 28 * i;
      int col = j;
      int idx = row + col;
      inputMat[0][idx] = (canvas.pixels[idx] >> 16) & 0xFF;
      inputMat[0][idx] = (float)inputMat[0][idx] / 255.0;
    }
  }
  canvas.endDraw();

  //network
  float[][] mat1 = multMat(inputMat, w1);
  mat1 = addMat(mat1, b1);
  relu(mat1);

  float[][] mat2 = multMat(mat1, w2);
  mat2 = addMat(mat2, b2);
  relu(mat2);

  float[][] mat3 = multMat(mat2, w3);
  mat3 = addMat(mat3, b3);

  //reshape to visualize
  float[][] reshapedMat1 = reshape(inputMat, 28);
  float[][] reshapedMat2 = reshape(mat1, 8);
  float[][] reshapedMat3 = reshape(mat2, 4);

  //visualization
  visualization.beginDraw();
  visualization.background(0);
  PVector[][] inputPos  = drawMat(reshapedMat1, 0, visualization);
  PVector[][] w1Pos     = drawMat(reshapedMat2, -100, visualization);
  PVector[][] w2Pos     = drawMat(reshapedMat3, -150, visualization);
  PVector[][] resultPos = drawResult(softmax(mat3), -200, visualization);

  if (visualizeLines) {
    drawLines(inputPos, w1Pos, visualization);
    drawLines(w1Pos, w2Pos, visualization);
    drawLines(w2Pos, resultPos, visualization);
  }
  visualization.endDraw();

  //result
  background(0);
  image(visualization, 0, 0);
  image(canvas, visualization.width, 0, visualization.width, visualization.height);
}

void keyPressed() {
  canvas.beginDraw();
  canvas.background(0);
  canvas.beginDraw();
}

float[][] softmax(float[][] x) {
  float[][] val = new float[1][x[0].length];
  float div = 0;
  for (int i = 0; i < x[0].length; i++) {
    float tempX = x[0][i];
    val[0][i] = exp(tempX);
    div += Math.exp(tempX);
  }
  println("");
  for (int i = 0; i < x[0].length; i++) {
    val[0][i] /= div;
    println("number", i, "=", Math.round(val[0][i]*100), "%");
  }
  return val;
}

float[][] loadMat(String fileName) {
  int row = 0, col = 0;

  String[] lines = loadStrings(fileName);
  col = lines.length;
  println("there are " + lines.length + " lines");

  if (lines.length > 0) {
    row = lines[0].split(",").length;
  } else {
    println("error!");
    return null;
  }

  float[][] mat = new float[col][row];

  for (int i = 0; i < col; i++) {
    //println(lines[i]);
    String[] linesInner = lines[i].split(",");
    for (int j = 0; j < row; j++) {
      mat[i][j] = float(linesInner[j]);
    }
  }
  return mat;
}

void relu(float[][] mat) {
  int col = mat.length;
  int row = mat[0].length;

  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      mat[i][j] = max(0, mat[i][j]);
    }
  }
}

void printMat(float[][] mat) {
  int row = mat[0].length;
  int col = mat.length;

  for (int i = 0; i < mat.length; i++) {
    print("|");
    for (int j = 0; j < mat[0].length; j++) {
      //print("i:" + i + " j:" + j + " ");
      print(mat[i][j] + "|");
    }
    print("\n");
  }
  println("row = " + row + " col = " + col);
}

float[][] multMat(float[][] matA, float[][] matB) {
  int rowA = matA.length;
  int colA = matA[0].length;
  int rowB = matB.length;
  int colB = matB[0].length;

  if (colA != rowB) {
    println("row col unmatch error!");    
    return null;
  }

  float[][] result = new float[rowA][colB];
  for (int i = 0; i < rowA; i++) {
    for (int j = 0; j < colB; j++) {
      for (int k = 0; k < colA; k++) {
        result[i][j] += matA[i][k] * matB[k][j];
      }
    }
  }
  return result;
}


float[][] addMat(float[][] matA, float[][] matB) {
  int rowA = matA.length;
  int colA = matA[0].length;
  int rowB = matB.length;
  int colB = matB[0].length;

  if (rowA != rowB || colA != colB) {
    print("shape unmatch error!");
    return null;
  }

  float[][] result = new float[rowA][colA];
  for (int i = 0; i < rowA; i++) {
    for (int j = 0; j < colA; j++) {
      result[i][j] = matA[i][j] + matB[i][j];
    }
  }
  return result;
}


PVector[][] drawMat(float[][] mat, float yPosition, PGraphics pg) {
  int row = mat.length;
  int col = mat[0].length;
  float scale = 12;
  float boxSize = 10;
  PVector[][] result = new PVector[row][col];

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      pg.pushMatrix();
      result[i][j] = new PVector(yPosition, i*scale - (row*scale) * 0.5, j*scale - (col*scale) * 0.5);
      pg.translate(result[i][j].x, result[i][j].y, result[i][j].z);
      pg.stroke(255);
      pg.fill(mat[i][j] * 255);
      pg.box(boxSize);
      pg.popMatrix();
    }
  }
  return result;
}

PVector[][] drawResult(float[][] mat, float yPosition, PGraphics pg) {
  int row = mat.length;
  int col = mat[0].length;
  float scale = 12;
  float boxSize = 10;
  PVector[][] result = new PVector[row][col];
  pg.textAlign(CENTER);
  pg.rectMode(CENTER);

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      color c = color(mat[i][j] * 255);
      pg.pushMatrix();
      result[i][j] = new PVector(yPosition, i*scale - (row*scale) * 0.5, j*scale - (col*scale) * 0.5);
      pg.translate(result[i][j].x, result[i][j].y, result[i][j].z);
      pg.rotateY(-HALF_PI);
      pg.stroke(255);
      pg.fill(c);
      pg.box(boxSize);
      pg.textSize(18);
      pg.fill(0);
      pg.noStroke();
      pg.rect(0, 10, 20, 20);
      pg.fill(c);
      pg.text(j, 0, 20, 10);
      pg.popMatrix();
    }
  }
  return result;
}

void drawLines(PVector[][] matAPos, PVector[][] matBPos, PGraphics pg) {
  int rowA = matAPos.length;
  int colA = matAPos[0].length;
  int rowB = matBPos.length;
  int colB = matBPos[0].length;

  for (int i = 0; i < rowA; i++) {  
    for (int j = 0; j < colA; j++) {
      for (int k = 0; k < rowB; k++) {  
        for (int l = 0; l < colB; l++) {
          pg.stroke(255, 100);
          pg.line(matAPos[i][j].x, matAPos[i][j].y, matAPos[i][j].z, matBPos[k][l].x, matBPos[k][l].y, matBPos[k][l].z);
        }
      }
    }
  }
}

//reshape 1d mat to 2d mat
float[][] reshape(float[][] mat, int desiredColNum) {
  int col = mat.length;
  int row = mat[0].length;
  col = desiredColNum;
  row = row/col;
  float[][] result = new float[desiredColNum][row];
  int idx = 0;
  if (row * col != mat[0].length) {
    println("reshape error");
    return null;
  }
  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      result[i][j] = mat[0][idx];
      idx++;
    }
  }
  return result;
}
