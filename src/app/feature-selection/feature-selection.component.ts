import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-feature-selection',
  templateUrl: './feature-selection.component.html',
  styleUrls: ['./feature-selection.component.scss']
})
export class FeatureSelectionComponent implements OnInit {
  treatedNaItems = {};
  objectKeys = Object.keys;
  variables: any = [];
  dataSource = {};
  constructor() { }

  ngOnInit() {
    this.getTreatedMissingValues();
    this.variables = [
      { column: 'Name' },
      { column: 'Age' },
      { column: 'Gender' },
      { column: 'Height' },
      { column: 'Weight' },
      { column: 'BMI' }
    ];
    this.dataSource = {
      chart: {
        caption: 'Countries With Most Oil Reserves [2017-18]',
        subCaption: 'In MMbbl = One Million barrels',
        xAxisName: 'Country',
        yAxisName: 'Reserves (MMbbl)',
        numberSuffix: 'K',
        theme: 'fusion'
      },
      data: [
        { label: 'Venezuela', value: '290' },
        { label: 'Saudi', value: '260' },
        { label: 'Canada', value: '180' },
        { label: 'Iran', value: '140' },
        { label: 'Russia', value: '115' },
        { label: 'UAE', value: '100' },
        { label: 'US', value: '30' },
        { label: 'China', value: '30' }
      ]
    };
  }

  getTreatedMissingValues() {
    this.treatedNaItems = {
      'columns': ['Name', 'Age', 'Gender', 'Height', 'Weight', 'BMI'],
      'treatment-type': ['mode', 'mean', 'mode', 'mean', 'mean', 'mean']
    }
  }
}
