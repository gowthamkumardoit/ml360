import { Component, OnInit } from '@angular/core';
import { FeatureSelectionService } from 'src/app/services/feature-selection.service';
import * as _ from 'lodash';
import { BehaviorSubject } from 'rxjs';
import { MatDialog } from '@angular/material';
import { SpinnerComponent } from 'src/app/shared/spinner/spinner.component';
import { FormGroup, FormControl, FormBuilder, Validators } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-feature-selection',
  templateUrl: './feature-selection.component.html',
  styleUrls: ['./feature-selection.component.scss']
})
export class FeatureSelectionComponent implements OnInit {
  treatedNaItems = {};
  objectKeys = Object.keys;
  variables: any = [];
  dataSource: any;
  selectedTargetVariable: any;
  selectedColumnforChart: any;
  boxplotValues: any;
  outlierValues: any;
  dataSourceForHistogram: any;
  histogramValues = [];
  naValuesTreatedColumns: any = [];
  naValuesTreatedValues: any = [];
  dataSourceOfBarChart: any;
  feature_columns: any;
  radioFormGroup: FormGroup;

  constructor(private featureSelectionService: FeatureSelectionService, public dialog: MatDialog, private fb: FormBuilder, private router: Router) { }

  ngOnInit() {
    const columns = JSON.parse(localStorage.getItem('selectedData'));
    let newColumns = [];
    if (columns) {
      columns['cols'].forEach((elem) => {
        newColumns.push({ 'column': elem });
      });
      this.variables = newColumns;
    }
    this.radioFormGroup = new FormGroup({
      variableType: new FormControl('', Validators.required)
    });
    this.radioFormGroup.setValue({ variableType: 'category' });
  }

  getMissingValues() {
    localStorage.setItem('targetColumn', this.selectedTargetVariable);
    let selectedFile = JSON.parse(localStorage.getItem('load_api_data'));
    console.log(this.radioFormGroup.value);
    const postData = { ...selectedFile, 'targetColumn': this.selectedTargetVariable, 'targetType': this.radioFormGroup.value.variableType };
    this.callSpinner();
    this.featureSelectionService.getMissingValues(postData).then((response: any) => {
      if (response) {
        let tempRes = response.treatedTypesList;
        let feature_columns_values = response.feature_columns_values;
        this.feature_columns = response.feature_columns;
        this.naValuesTreatedColumns = [];
        this.naValuesTreatedValues = [];

        this.featureSelectionService.dragAndDrop.next({ 'original': this.variables, 'featured': this.feature_columns });
        let key;
        tempRes.forEach((elem, i) => {
          key = Object.keys(elem);
          this.naValuesTreatedColumns.push(Object.keys(elem));
          this.naValuesTreatedValues.push(elem[key]);
        });

        let tempValues = feature_columns_values.map((ele, i) => {
          return { "label": ele.Variables, "value": ele.Importance * 100 };
        });
        this.getTreatedMissingValues();
        this.prepareBarChart(tempValues);
        this.dialog.closeAll();
      }
    }).catch((err) => {
      console.log(err);
    })
  }

  callSpinner() {
    this.dialog.open(SpinnerComponent, { disableClose: true });
  }
  getTreatedMissingValues() {
    this.treatedNaItems = {
      'columns': this.naValuesTreatedColumns,
      'treatment-type': this.naValuesTreatedValues
    }
  }

  startML() {
    this.router.navigate(['/result']);
    const variableType = this.radioFormGroup.value.variableType;
    localStorage.setItem('variableType', variableType);

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
        "showHoverEffect": "1"
      },
      "data": this.histogramValues
    }
  }

  prepareBarChart(data) {
    this.dataSourceOfBarChart = {
      "chart": {
        "theme": "fusion",
        "caption": "Feature Importance",
        "subCaption": "(In Percentage)",
        "yAxisName": "Variables",
        "xAxisName": "Relative Importance",
        "numberSuffix": "%",
        "alignCaptionWithCanvas": "0"
      },
      "data": data
    }
  }
}
