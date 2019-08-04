import { Component, OnInit, Inject } from '@angular/core';
import { MatBottomSheetRef, MAT_BOTTOM_SHEET_DATA } from '@angular/material/bottom-sheet';
import { Observable, BehaviorSubject } from 'rxjs';

@Component({
  selector: 'app-bottom-sheet',
  templateUrl: './bottom-sheet.component.html',
  styleUrls: ['./bottom-sheet.component.scss']
})
export class BottomSheetComponent implements OnInit {
  progress: any;
  dataProgress: any;
  isFileUploaded = false;
  constructor(private bottomSheetRef: MatBottomSheetRef<BottomSheetComponent>, @Inject(MAT_BOTTOM_SHEET_DATA) public data: any) {
    this.dataProgress = data;
    // data.progress.subscribe((item) => {
    //   console.log('view++++', this.progress);
    //   this.progressValue.next(item);
    // });
  }

  openLink(event: MouseEvent): void {
    this.bottomSheetRef.dismiss();
    event.preventDefault();
  }

  ngOnInit() {
    this.dataProgress.progress.subscribe((item) => {
      this.progress = item;
      console.log('out---', (parseInt(this.progress, 10) === 100));
      if (parseInt(this.progress, 10) === 100) {
        console.log((parseInt(this.progress, 10) === 100));
        this.isFileUploaded = true;
        return;
      }
    });
  }


}
