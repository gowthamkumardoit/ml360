import { Component, OnInit } from '@angular/core';
import { FeatureSelectionService } from 'src/app/services/feature-selection.service';
import * as _ from 'lodash';

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
  selectedTargetVariable: any;
  selectedColumnforChart: any;
  boxplotValues: any;
  outlierValues: any;
  dataSourceForHistogram = {};
  histogramValues = [];
  naValuesTreatedColumns: any = [];
  naValuesTreatedValues: any = [];
  constructor(private featureSelectionService: FeatureSelectionService) { }

  ngOnInit() {
    const columns = JSON.parse(localStorage.getItem('selectedData'));
    let newColumns = [];
    if (columns) {
      columns['cols'].forEach((elem) => {
        newColumns.push({ 'column': elem });
      });
      this.variables = newColumns;
    }
  }

  getMissingValues() {
    localStorage.setItem('targetColumn', this.selectedTargetVariable);
    let selectedFile = JSON.parse(localStorage.getItem('load_api_data'));
    const postData = { ...selectedFile, 'targetColumn': this.selectedTargetVariable };
    this.featureSelectionService.getMissingValues(postData).then((response: any) => {
      if(response) {
        let tempRes = response.result;
        this.naValuesTreatedColumns = [];
        this.naValuesTreatedValues = [];
        let key;
        tempRes.forEach((elem, i) => {
          key = Object.keys(elem);
          this.naValuesTreatedColumns.push(Object.keys(elem));
          this.naValuesTreatedValues.push(elem[key]);
        });
        this.getTreatedMissingValues();
      }
    }).catch((err) => {
      console.log(err);
    })
  }


  getTreatedMissingValues() {
    this.treatedNaItems = {
      'columns': this.naValuesTreatedColumns,
      'treatment-type': this.naValuesTreatedValues
    }
  }

  loadChart() {
    localStorage.setItem('selectedColumn', this.selectedColumnforChart);
    let selectedFile = JSON.parse(localStorage.getItem('load_api_data'));
    const postData = { ...selectedFile, 'chart_column': this.selectedColumnforChart };

    this.featureSelectionService.loadChart(postData).then((res) => {
      let tempBoxValues = res['columns'];
      this.outlierValues = res['outliers'];
      this.boxplotValues = _.difference(tempBoxValues, this.outlierValues);
      this.histogramValues = [];
      tempBoxValues.forEach((ele) => {
        this.histogramValues.push({ value: ele });
      });
      this.prepareChart();
      this.prepareHistogram();
    }).catch((err) => {
      console.log(err);
    })
  }

  prepareChart() {
    this.dataSource =
      {
        "chart": {
          "theme": "fusion",
          "caption": "Distribution of " + this.selectedColumnforChart,
          "legendBorderAlpha": "0",
          "legendShadow": "0",
          "legendPosition": "right",
          "showValues": "0",
          "toolTipColor": "#ffffff",
          "toolTipBorderThickness": "0",
          "toolTipBgColor": "#000000",
          "toolTipBgAlpha": "80",
          "toolTipBorderRadius": "2",
          "toolTipPadding": "5",
          "showAllOutliers": '1'
        },

        "categories": [
          {
            "category": [
              {
                "label": this.selectedColumnforChart
              },

            ]
          }
        ],
        "dataset": [
          {
            "seriesname": this.selectedColumnforChart,
            "lowerBoxColor": "#0075c2",
            "upperBoxColor": "#1aaf5d",
            "data": [
              {
                "value": (this.boxplotValues).toString(),
                "outliers": (this.outlierValues).toString()
              }

            ],

          }
        ]

      };
  }

  prepareHistogram() {
    this.dataSourceForHistogram = {
      "chart": {
        "theme": "fusion",
        "caption": "Sales Trends",
        "subcaption": "2016 - 2017",
        "xaxisname": "Month",
        "yaxisname": "Revenue",
        "showvalues": "0",
        "numberprefix": "$",
        // "numVisiblePlot": "12",
        // "scrollheight": "10",
        // "flatScrollBars": "1",
        // "scrollShowButtons": "0",
        // "scrollColor": "#cccccc",
        "showHoverEffect": "1"
      },

      "data": this.histogramValues

    }
    console.log(this.dataSourceForHistogram);
  }
}
