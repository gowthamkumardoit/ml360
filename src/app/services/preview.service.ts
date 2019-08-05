import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { MatSnackBar } from '@angular/material';
import { AngularFirestore, AngularFirestoreDocument } from '@angular/fire/firestore';
import { AngularFireAuth } from '@angular/fire/auth';
import { AuthService } from './auth.service';
import { take, map } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';
import { URL, PORT } from '../constant/app.constants';
@Injectable({
    providedIn: 'root'
})
export class PreviewService {
    user;
    url = URL;
    port = PORT;
    fileListForUser: any[] = [];
    constructor(private snackBar: MatSnackBar, private db: AngularFirestore, private afauth: AngularFireAuth, private http: HttpClient) {
        this.user = JSON.parse(localStorage.getItem('user'));
    }
    uploadFilesRef = this.db.collection('uploadFiles').ref;

    getFilesForUsers() {

        return this.db.collection('uploadFiles').snapshotChanges().pipe(
            map((actions) => {
                return actions.map(a => {
                    const data = a.payload.doc.data();
                    const id = a.payload.doc.id;
                    return { docId: id, ...data };
                });
            }),
            map((files) => {
                const filterdFiles = files.filter((element: any, i: number) => {
                    return (element.id === this.afauth.auth.currentUser.uid);
                });
                return filterdFiles;
            })
        );
    }

   

    getDownloadURLs(data) {
        const docRef = this.db.collection('downloadurls').doc(data.id).ref;
        let downloadObj = [];
        let individualDocs = [];
        docRef.get().then(function (doc) {
            if (doc.exists) {
                downloadObj = doc.data()['url'].filter((item) => {
                    return Object.keys(item) == data['docId']
                });
                downloadObj.filter((item) => {
                    if (Object.keys(item) == data['docId']) {
                        individualDocs.push(item[data['docId']])
                    }
                });
            } else {
                // doc.data() will be undefined in this case
                console.log("No such document!");
            }
        }).then(() => {
            // let splitData = individualDocs[0].split("token=")[0] || '';
            this.passFileFromFirebasetoBackend({downloadURL: individualDocs[0], ...data});
        }).catch(function (error) {
            console.log("Error getting document:", error);
        });
    }

    passFileFromFirebasetoBackend(data) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api`, data).subscribe((res) => {
                console.log('response');
                resolve(true);
            });
        });
    }
}

// /vfWusrIPy6XsocDOYGoiCDSzd3g1/Off Campus_Report Updated_1.xlsx