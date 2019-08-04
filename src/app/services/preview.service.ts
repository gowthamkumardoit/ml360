import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { MatSnackBar } from '@angular/material';
import { AngularFirestore, AngularFirestoreDocument } from '@angular/fire/firestore';
import { AngularFireAuth } from '@angular/fire/auth';
import { AuthService } from './auth.service';
import { take, map } from 'rxjs/operators';
@Injectable({
    providedIn: 'root'
})
export class PreviewService {
    user;
    fileListForUser: any[] = [];
    constructor(private snackBar: MatSnackBar, private db: AngularFirestore, private afauth: AngularFireAuth) {
        this.user = JSON.parse(localStorage.getItem('user'));
    }
    uploadFilesRef = this.db.collection('uploadFiles').ref;

    getFilesForUsers() {
        return this.db.collection('uploadFiles').valueChanges().pipe(
            map((files) => {
                const filterdFiles = files.filter((element: any, i: number) => {
                    return (element.id === this.afauth.auth.currentUser.uid);
                });
                return filterdFiles;
            })
        );
    }
}
