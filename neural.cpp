#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

template <typename T>
class Neural{
private:

  vector <T> input;
  vector <T> answer;
  vector<vector <T>> WeightOld; //старые веса
  vector<vector <T>> Weight; //веса
  vector<vector <T>>Output; //рез-ты ф-ии активации нейронов
  bool Function = false; //функция активации 0-сигмойд, 1-гип.тангенс

  int length; //длина скрытых нейронов
  int height; //высота скрытых нейронов
  int SizeInput; //размер входных данных
  int SizeOutput; //размер выходных данных
  double GRADw =0;//градиент веса
  double E;// скорость обучения
  double a = 0;//момент
  vector <T> Error;

  void Calculate(){//вычисление ответа
    int temp=0;
    int WeightNum=0;
    //очистка
    for(auto& i: Output){
      for(auto& j: i){
        j=0;
      }
    }

      for(auto& i: Output){
        for(auto& j: i){
            if(temp){
              for(auto&k: Output[temp-1]){
              j+=k*Weight[temp][WeightNum];
              WeightNum++;
            }
            }else{
              for(auto&k: input){
              j+=k*Weight[temp][WeightNum];
              WeightNum++;
              }
            }
            j=Activation(j);
          }
          WeightNum=0;
          temp++;
        }
        temp=0;
    }


  double Sigmoid(T& h){ //функция активации только с положительными числами
    h = 1/(1+exp(-h));
    return h;
  }

  double Tanh(T& h){ //функция активации с отрицательными числами
    h = tanh(h);
    return h;
  }

  double Activation(T& h)
  {
    if(!Function){
      return Sigmoid(h);
    }else{
      return Tanh(h);
    }
  }

  T Derivative(T& h){ //Производная
    if(!Function){
      return (1-h)*h;
    }else{
      return (1/(0.5*(cosh(2*h)+1))); //добавить функцию
    }
  }

public:
  Neural(){

  }
  Neural(vector<T>Input, int Length, int Height, double Ee, bool Funct, int Sizeout){ //входные данные, длина, высота,
    //скорость обучения, функция активации, размер выходных данных
    input = Input;
    length = Length;
    height = Height;
    Function = Funct;
    SizeOutput = Sizeout;
    E= Ee;

    for(int i=0; i < length; i++){
      Weight.push_back(vector<T>());
      WeightOld.push_back(vector<T>());
      if(i<length-1 && i > 0){
        for(int j=0; j < height*height; j++){
          Weight[i].push_back(rand()%100);
          Weight[i][j] /=100;
          WeightOld[i].push_back(0);
  //        cout << "центральный вес: " << Weight[i][j] << endl;
          }
        }else if (i ==length-1){
          for(int j=0; j < height*SizeOutput; j++){
            Weight[i].push_back(rand()%100);
            Weight[i][j] /=100;
            WeightOld[i].push_back(0);
      //      cout << "последний вес: " << Weight[i][j] << endl;
          }
        } else if (i == 0){
          for(int j=0; j < input.size()*height; j++){
            Weight[i].push_back(rand()%100);
            Weight[i][j] /=100;
            WeightOld[i].push_back(0);
    //        cout << "вес входа: " << Weight[i][j] << endl;
          }
        }
    }

     for(int i=0; i < length; i++){
       Output.push_back(vector<T>());
       if(i<length-1){
         for(int j=0; j < height;j++){
           Output[i].push_back(0);

            }
        } else{
          for(int j=0; j < SizeOutput;j++){
           Output[i].push_back(0);
             }
        }
      }
  }

  void SetParam(vector <T> Input) //установить вводные пданные
  {
    input = Input;

    Calculate();
  }

  vector <T> GetOutput(){ //получить ответ
    return Output[length-1];
  }

  vector<T> GetError(vector<double> d){ //вычислить ошибку
    Error.assign(d.size(),0);
    for(int i=0; i< d.size(); i ++){
    Error[i] = pow(d[i]-GetOutput()[i],2);
    }
    return Error;

  }
  void Learning(vector<T> Answer){
    int temp=length-1;
    int temp2=0;
    double err =0;
    int WeightNum=0;

      for(int i = temp; i >= -1; i--){//реверсивный переход между слоями нейронов
        if(i>=0){
        for(auto& j: Output[i]){
            if(i<temp){
              for(auto&k: Output[i+1]){ // ПРОБЛЕМА ЗСЬ!
              GRADw= k*j;
              Weight[i+1][temp2+Output[i+1].size()*WeightNum]=Weight[i+1][temp2+Output[i+1].size()*WeightNum]-E*GRADw-a*WeightOld[i+1][temp2+height*WeightNum]; //новые веса двух нейронов
              err+= Weight[i+1][temp2+Output[i+1].size()*WeightNum]*k; //рез-ты функции активации скрытых нейронов, т.е. дельта для H
              WeightOld[i+1][temp2+height*WeightNum] = Weight[i+1][temp2+height*WeightNum];

    //            cout << "скорректированные веса (конец):  " << Weight[i+1][temp2+Output[i+1].size()*WeightNum] << endl;
    //          cout << "Итератор веса:"<<temp2+Output[i+1].size()*WeightNum << endl;
              WeightNum++;
            }
            }else if(i == temp){
              /*for(auto&k: Answer){
              err+=(j-k);
            }*/
            err=j-Answer[temp2];
            }
            j = err*Derivative(j);

            err = 0;
            temp2++;
            WeightNum=0;
          }
          temp2=0;


        }else if(i < 0){
            for(int j = 0; j < height;j++){
          for (int k=0; k < input.size();k++)
          {
            GRADw=input[k]*Output[0][j];
            Weight[0][WeightNum]= Weight[0][WeightNum]-E*GRADw-a*WeightOld[0][WeightNum]; //новые веса двух нейронов
            WeightOld[0][WeightNum] = Weight[0][WeightNum];
//              cout << "скорректированные веса (ввода):  " << Weight[0][WeightNum] << endl;
    //          cout << "Итератор веса:"<< WeightNum << endl;
            WeightNum++;

          }}
          WeightNum=0;
          }

        }
  }

  void Training(vector<vector <T>> InputData, vector <vector<T>> AnswerData, int Epoch){
    int Var = AnswerData.size();

    for (int i = 0; i < Epoch; i++){
      for (int j = 0; j < Var; j++){
        SetParam(InputData[j]);
        cout << GetError(AnswerData[j])[0]*100 << "%" << endl;
        Learning(AnswerData[j]);
      }
      cout << endl;
    }
  }
};
