import { Component, OnInit, Input } from '@angular/core';
import { DataSource } from '@angular/cdk/table';

@Component({
  selector: 'app-box-plot',
  templateUrl: './box-plot.component.html',
  styleUrls: ['./box-plot.component.scss']
})
export class BoxPlotComponent implements OnInit {
  @Input('dataSource') dataSource; 
  constructor() { }

  ngOnInit() {
  }

}
