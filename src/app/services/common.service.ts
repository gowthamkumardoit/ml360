import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { MatSnackBar } from '@angular/material';
import { AngularFirestore, AngularFirestoreDocument } from '@angular/fire/firestore';
import { take, map } from 'rxjs/operators';
import { AngularFireAuth } from '@angular/fire/auth';
@Injectable({
  providedIn: 'root'
})
export class CommonService {
  loggedIn = new BehaviorSubject<boolean>(false);
  users;
  constructor(private snackBar: MatSnackBar, private db: AngularFirestore, private afauth: AngularFireAuth) {
    this.users = JSON.parse(localStorage.getItem('user'));

  }

  userRef = this.db.collection('uploadFiles').ref;
  getCurrentLoggedInUser() {
    return this.db.collection('users').valueChanges().pipe(
      map((user) => {
      })
    );
  }

  //add download url to the file uploaded account
  addDownloadURL(id) {
    const downloadURLref: AngularFirestoreDocument = this.db.doc(`downloadurls/${id}`);
    downloadURLref.set({
      url: []
    });
  }

}
