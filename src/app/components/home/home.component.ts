import { Component, OnInit } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { AngularFireStorage } from '@angular/fire/storage';
import { AngularFirestore } from '@angular/fire/firestore';
import { CommonService } from '../../services/common.service';
import { FormControl, Validators } from '@angular/forms';
import { Observable } from 'rxjs';
import { Animations } from '../../animations/fadein-fadeout.animation';
import { MatBottomSheet } from '@angular/material/bottom-sheet';
import { BottomSheetComponent } from '../../shared/bottom-sheet/bottom-sheet.component';
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
  animations: [Animations.animeTrigger]
})
export class HomeComponent implements OnInit {
  uploadProgress: Observable<number>;
  fileName: string = 'Select the file!';
  isFileupload: boolean = false;
  isNextEnabled: boolean = false;
  uid: string;
  files: FileList;
  selected: any;
  isSelected: boolean;
  items = [];
  isFileUploadedtoDb = false;
  isFileSubmitted = false;
  progressValue = 20;

  uploadFormControl = new FormControl();
  formValues = [
    { name: 'pipe', value: 'Pipe (|)' },
    { name: 'comma', value: 'Comma (,)' },
    { name: 'tab', value: 'Tab (\t)' },
    { name: 'semicolon', value: 'Semi colon (;)' },
  ];

  constructor(private afauth: AngularFireAuth, private storage: AngularFireStorage, private db: AngularFirestore, private commonService: CommonService, private bottomSheet: MatBottomSheet) {
    this.items = [
      { step: '1', name: 'Uploading the file...' },
      { step: '2', name: 'Finding the missing values...' },
      { step: '3', name: 'Treating all missing values...' },
      { step: '4', name: 'Running the Random Forest Algorthim...' },
      { step: '5', name: 'Extracting the feature columns..' }
    ];
  }

  ngOnInit() {
  }

  selectDelimiter() {
    if (this.selected != null) {
      this.isSelected = true;
    } else {
      this.isSelected = false;
    }

  }

  changeAttr(event) {
    this.fileName = event.target.files[0].name || 'Select the file!';
    this.isFileupload = true;
    this.files = event.target.files[0];
    this.uid = this.afauth.auth.currentUser.uid;
  }

  submit() {
    if (!this.files) {
      this.commonService.showError('File should be uploaded');
      return;
    }

    if (this.uploadFormControl.hasError('required')) {
      this.commonService.showError('Please select the delimiter');
      return;
    }

    this.uploadFile(this.files, this.uid, this.uploadFormControl.value);
  }

  uploadFile(files, uid, delim) {

    const randomId = 'file_' + Math.floor(Math.random() * 1000000);
    const filePath = 'uploads/' + randomId;
    const uploadTask = this.storage.upload(filePath, files);
    this.isFileSubmitted = true;
    this.uploadProgress = uploadTask.percentageChanges();
    this.bottomSheet.open(BottomSheetComponent, { disableClose: true, data: { progress:  this.uploadProgress } });
    
    this.uploadProgress.subscribe((progress) => {

     
      if (progress > 20) {
        this.progressValue = progress;
      }
      

      if (progress === 100) {
        this.isFileUploadedtoDb = true;
        this.isFileSubmitted = false;
      }
    });
   
    uploadTask.then((data) => {
      this.updateDb(uid, randomId, delim);
    });

  }

  updateDb(uid, randomId, delim) {
    this.db.collection('uploadFiles').add({
      name: randomId,
      id: uid,
      delimiter: delim.name
    }).then((data) => {
      console.log('data updated', data);
    });
  }


}
