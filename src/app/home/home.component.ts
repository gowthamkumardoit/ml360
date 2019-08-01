import { Component, OnInit } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { AngularFireStorage } from '@angular/fire/storage';
import { AngularFirestore } from '@angular/fire/firestore';
import { CommonService } from '../services/common.service';
import { FormControl, Validators } from '@angular/forms';
import { Upload } from '../models/interface';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  fileName: string = 'Select the file!';
  isFileupload: boolean = false;
  isNextEnabled: boolean = false;
  uid: string;
  files: FileList;
  selected: any;
  isSelected: boolean;

  uploadFormControl = new FormControl();
  formValues: Upload[] = [
    { name: 'pipe', value: 'Pipe (|)' },
    { name: 'comma', value: 'Comma (,)' },
    { name: 'tab', value: 'Tab (\t)' },
    { name: 'semicolon', value: 'Semi colon (;)' },
  ];

  constructor(private afauth: AngularFireAuth, private storage: AngularFireStorage,
    private db: AngularFirestore, private commonService: CommonService) { }

  ngOnInit() {
  }

  selectDelimiter() {
    if (this.selected != null) {
      this.isSelected = true;
    } else {
      this.isSelected = false;
    }
    console.log('isSelected', this.isSelected);
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
