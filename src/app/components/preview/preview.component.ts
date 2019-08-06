import { Component, OnInit } from '@angular/core';
import { PreviewService } from 'src/app/services/preview.service';
import { AuthService } from 'src/app/services/auth.service';
import { FormControl, Validators } from '@angular/forms';
import { FileInterface } from '../../interfaces/file';
import {MatBottomSheet, MatBottomSheetRef} from '@angular/material/bottom-sheet';
import { BottomSheetComponent } from 'src/app/shared/bottom-sheet/bottom-sheet.component';
@Component({
  selector: 'app-preview',
  templateUrl: './preview.component.html',
  styleUrls: ['./preview.component.scss']
})
export class PreviewComponent implements OnInit {
  rows: any[] = [];
  cols: any[] = [];
  describeAttributes: any = [];
  describeRows: any = {};
  percentageOfNA: any = {};
  skewAndKurtosis: any = {};
  fileControl = new FormControl('', [Validators.required]);
  filesAvailable;
  isPreviewAvailable: boolean;
  constructor(private previewService: PreviewService, private authService: AuthService, private _bottomSheet: MatBottomSheet) {
    this.isPreviewAvailable = false;
  }

  ngOnInit() {

    // this.getRows();
    this.getDescribeRows();
    this.getPercentageOfNA();
    this.getSkewandKurtosis();
    this.getFilesForUsers();
  }

  getRows() {
    this.rows = [
      ['Name', 'Age', 'Gender', 'Height', 'Weight', 'BMI'],
      ['Gowtham', 29, 'Male', 160, 65, 24],
      ['Bala', 34, 'Male', 175, 75, 29],
      ['Abishek', 21, 'Male', 179, 85, 30],
      ['Karan', 40, 'Male', 180, 65, 27],
      ['Ashok', 33, 'Male', 184, 95, 28]
    ];
  }

  getDescribeRows() {
    this.describeRows = {
      columns: ['age', 'height', 'weight', 'bmi'],
      rows: [
        { count: [5.00000, 5.00000, 5.0000, 5.0000] },
        { mean: [5.00000, 5.00000, 5.0000, 5.0000] },
        { std: [5.00000, 5.00000, 5.0000, 5.0000] },
        { min: [5.00000, 5.00000, 5.0000, 5.0000] },
        { '25%': [5.00000, 5.00000, 5.0000, 5.0000] },
        { '50%': [5.00000, 5.00000, 5.0000, 5.0000] },
        { '75%': [5.00000, 5.00000, 5.0000, 5.0000] },
        { max: [5.00000, 5.00000, 5.0000, 5.0000] },
      ],
    };
  }

  getPercentageOfNA() {
    this.percentageOfNA = {
      columns: ['name', 'age', 'gender', 'height', 'weight', 'bmi'],
      rows: [
        { count: [0, 5, 1, 0, 0, 0] },
        { percentage: [0, 100, 20, 0, 0, 0] }
      ],
    };
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
      this.previewService.getDownloadURLs(this.fileControl.value).then((response: any) => {
        console.log('respone in component', response);
        if (response) {
          this.isPreviewAvailable = true;
          this.cols = Array(response.cols);
          this.rows = Array(response.rows);
          console.log(this.rows);
        }
      });
    }
  }
}
