import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.scss']
})
export class ResultComponent implements OnInit {

  algorithmArray: any = [];
  equation: any = [];
  finalEquation: any;
  constructor() { }

  ngOnInit() {
    this.algorithmArray = [
      { title: 'Linear', subtitle: 'Regression based algorithm', rmse: 13.222, r_square: 87.456, adj_r_square: 86.23 },
      { title: 'KNN', subtitle: 'Regression based algorithm', rmse: 11.222, r_square: 89.456, adj_r_square: 88.23 },
      { title: 'Random Forest', subtitle: 'Regression based algorithm', rmse: 10.222, r_square: 90.111, adj_r_square: 90.012 },
    ];

    this.equation = [
      { col: 'Age', co_ef: '20.33' },
      { col: 'Height', co_ef: '12.65' },
      { col: 'Weight', co_ef: '3.81' },
      { col: 'BMI', co_ef: '7.10' }
    ];

    this.finalEquation = this.equation.map((data) => {
      return ' ' + data.co_ef + '';
    });
  }

}

