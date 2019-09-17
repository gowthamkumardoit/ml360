import { Component, OnInit } from '@angular/core';
import { FeatureSelectionService } from 'src/app/services/feature-selection.service';
import { SpinnerComponent } from 'src/app/shared/spinner/spinner.component';
import { MatDialog } from '@angular/material';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.scss']
})
export class ResultComponent implements OnInit {

  algorithmArray: any = [];
  equation: any = [];
  finalEquation: any;
  variableType;
  constructor(private featureSelectionService: FeatureSelectionService, private dialog: MatDialog) { }

  ngOnInit() {
    setTimeout(() => {
      this.executeAlgorithms();
    }, 2000)
  }
  executeAlgorithms() {
    this.variableType = localStorage.getItem('variableType');
    let targetColumn = localStorage.getItem('targetColumn');
    let obj = { 'target': targetColumn };
    this.algorithmArray = [];
    if (this.variableType == 'numeric') {
      this.callSpinner();
      this.featureSelectionService.executeLinearRegressionAlgorithm(obj).then(
        (res) => {
          this.algorithmArray.push(res);
          this.featureSelectionService.executeRandomForestRegressionAlgorithm(obj).then(
            (res) => {
              this.algorithmArray.push(res);
              this.featureSelectionService.executeKNNRegressionAlgorithm(obj).then(
                (res) => {
                  this.algorithmArray.push(res);
                  this.dialog.closeAll();
                }
              )
            }
          )
        });
    } else if (this.variableType == 'category') {
      this.callSpinner();
      this.featureSelectionService.executeLogisticClassifierAlgorithm(obj).then(
        (res) => {
          this.algorithmArray.push(res);
          this.featureSelectionService.executeRandomForestClassifierAlgorithm(obj).then(
            (res) => {
              this.algorithmArray.push(res);
              this.featureSelectionService.executeGradientBoostClassifierAlgorithm(obj).then(
                (res) => {
                  this.algorithmArray.push(res);
                  this.dialog.closeAll();
                }
              )
            }
          )
        });
    }
  }
  callSpinner() {
    this.dialog.open(SpinnerComponent, { disableClose: true });
  }
}

