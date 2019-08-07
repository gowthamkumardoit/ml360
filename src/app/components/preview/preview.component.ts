import { Component, OnInit } from '@angular/core';
import { PreviewService } from 'src/app/services/preview.service';
import { AuthService } from 'src/app/services/auth.service';
import { FormControl, Validators } from '@angular/forms';
import { FileInterface } from '../../interfaces/file';
import { MatBottomSheet, MatBottomSheetRef } from '@angular/material/bottom-sheet';
import { MatDialog, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { SpinnerComponent } from 'src/app/shared/spinner/spinner.component';

@Component({
  selector: 'app-preview',
  templateUrl: './preview.component.html',
  styleUrls: ['./preview.component.scss']
})
export class PreviewComponent implements OnInit {
  rows: any[] = [];
  cols: any[] = [];
  summaryRows: any[] = [];
  summaryCols: any[] = [];
  describeAttributes: any = [];
  describeRows: any = {};
  percentageOfNA: any = {};
  skewAndKurtosis: any = {};
  fileControl = new FormControl('', [Validators.required]);
  filesAvailable;
  isFileSelected: boolean;
  width = 900;
  height = 600;
  type = 'scrollcombidy2d';
  dataFormat = 'json';
  dataSource = null;
  naChartColumns: any[] = [];
  naCountRows = [];
  naPercentRows = [];
  skewness: any;
  kurtosis: any;
  skewnessAndKurtosisCols: any;
  constructor(private previewService: PreviewService, private authService: AuthService, public dialog: MatDialog) {
    this.isFileSelected = false;
  }

  ngOnInit() {

    this.getSkewandKurtosis();
    this.getFilesForUsers();
  }

  getSkewandKurtosis() {
    this.skewAndKurtosis = {
      columns: ['age', 'height', 'weight', 'bmi'],
      rows: [
        { skewness: [5, 1, 0, 0] },
        { kurtosis: [100, 20, 0, 0] }
      ],
    };
  }


  getFilesForUsers() {
    this.previewService.getFilesForUsers().subscribe((data) => {
      console.log(data);
      this.filesAvailable = data;
    });

  }

  fileSelectionEvent() {

    if (this.fileControl && this.fileControl.value) {
      this.callSpinner();
      this.previewService.getDownloadURLs(this.fileControl.value).then((response: any) => {
        if (response) {
          this.naChartColumns = [];
          this.naCountRows = [];
          this.naPercentRows = [];
          this.cols = Array(response.cols);
          this.rows = Array(response.rows);
          this.summaryCols = Array(response.summary_cols);
          this.summaryRows = Array(response.summary_rows);
          response.cols.forEach((elem: any, i: number) => {
            this.naChartColumns.push({ label: elem });
          });
          response.na_data_rows.forEach((elem: any, i: number) => {
            this.naCountRows.push({ value: elem.count_of_missing_values });
            this.naPercentRows.push({ value: elem.percent_of_missing_values });
          });
          this.skewness = response.skew;
          this.kurtosis = response.kurtosis;
          this.skewnessAndKurtosisCols = Object.keys(response.skew);
          const yMax = response.yMax;
          this.getChart(yMax);
          this.isFileSelected = true;
          this.dialog.closeAll();
        }
      });
    }
  }

  callSpinner() {
    this.dialog.open(SpinnerComponent, { disableClose: true });
  }

  getChart(yMax) {
    const data = {
      chart: {
        caption: 'NA - values in count and percentage',
        drawcrossline: '1',
        yaxisname: 'NA values in count',
        syaxisname: 'NA values in percentage',
        showvalues: '0',
        labeldisplay: 'rotate',
        plothighlighteffect: 'fadeout',
        theme: 'fusion',
        plotSpacePercent: 10,
        numVisiblePlot: '20',
        scrollheight: '10',
        flatScrollBars: '1',
        scrollShowButtons: '0',
        scrollColor: '#cccccc',
        showHoverEffect: '1',
        numDivLines: '5',
        sYAxisMaxValue: '100',
        pYAxisMaxValue: yMax,
      },
      categories: [
        {
          category: this.naChartColumns
        }
      ],
      dataset: [
        {
          seriesname: 'Count of NA',
          showvalues: '0',
          plottooltext: 'Count of NA in $label : <b>$dataValue</b>',
          data: this.naCountRows
        },

        {
          seriesname: 'Percentage of NA',
          parentyaxis: 'S',
          renderas: 'line',
          showvalues: '0',
          plottooltext: 'Percentage of NA in $label : <b>$dataValue</b>%',
          data: this.naPercentRows
        }
      ]
    };

    this.dataSource = data;
  }
}


