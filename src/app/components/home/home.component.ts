import { Component, OnInit } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { AngularFireStorage } from '@angular/fire/storage';
import { CommonService } from '../../services/common.service';
import { FormControl } from '@angular/forms';
import { Observable } from 'rxjs';
import { Animations } from '../../animations/fadein-fadeout.animation';
import { Router } from '@angular/router';
import { AlertsService } from 'src/app/services/alert.service';
import { HomeService } from 'src/app/services/home.service';
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
  animations: [Animations.animeTrigger]
})
export class HomeComponent implements OnInit {
  uploadProgress: Observable<number>;
  fileName = 'Select the file!';
  isFileupload = false;
  isNextEnabled = false;
  user: any;
  files: FileList;
  selected: any;
  isSelected: boolean;
  items = [];
  isFileUploadedtoDb = false;
  isFileSubmitted = false;
  progressValue = 20;
  isFileDeleted = false;
  uploadTask: any;

  uploadFormControl = new FormControl();
  formValues = [
    { name: 'pipe', value: 'Pipe (|)' },
    { name: 'comma', value: 'Comma (,)' },
    { name: 'tab', value: 'Tab (\t)' },
    { name: 'semicolon', value: 'Semi colon (;)' },
  ];

  constructor(private afauth: AngularFireAuth, private storage: AngularFireStorage, private commonService: CommonService, private router: Router,
    private alertsService: AlertsService, private homeService: HomeService) {

  }

  ngOnInit() {
  }

  // Selecting the delimiter of the uploaded file.
  selectDelimiter() {
    if (this.selected != null) {
      this.isSelected = true;
    } else {
      this.isSelected = false;
    }
  }

  // This change event will trigger when the user select the file and click OK or Cancel button in the dialog box.
  changeAttr(event) {
    this.fileName = event.target.files[0].name || 'Select the file!';
    this.isFileupload = true;
    this.files = event.target.files[0];
    this.user = JSON.parse(localStorage.getItem('user'));
  }

  // After Selecting the file and delimiter submit function will trigger by clicking submit button.
  submit() {

    if (!this.files) {
      this.alertsService.showError('File should be uploaded');
      return;
    }

    if (this.uploadFormControl.hasError('required')) {
      this.alertsService.showError('Please select the delimiter');
      return;
    }

    this.uploadFile(this.files);
    this.isFileDeleted = false;
  }

  // Uploading the files to the Firebase Storage
  uploadFile(files) {
    const filePath = `uploads/${this.user.uid}/ ${this.fileName}`;
    this.uploadTask = this.storage.upload(filePath, files);
    this.isFileSubmitted = true;
    this.uploadProgress = this.uploadTask.percentageChanges();
    this.uploadProgress.subscribe((progress) => {
      this.progressValue = progress;

      if (progress === 100) {
        this.isFileUploadedtoDb = true;
        this.isFileSubmitted = false;
        this.deleteFile();
      }
    });
  }

  // checks the file name is already exists in the firestore, if yes, delete the entry and update it or just add a new entry.
  deleteFile() {
    this.homeService.deleteFile(this.fileName, this.user.uid).then((res) => {
      this.uploadTask.then((data) => {
        this.updateDb();
      });
    });
  }

  // Uploading the file details to the Firebase Database for logged in user
  updateDb() {
    this.homeService.updateDb(this.user.uid, this.uploadFormControl.value, this.fileName).then((res) => {
      if (res) {
        this.router.navigate(['preview']);
      }
    });
  }
}
