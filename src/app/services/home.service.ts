import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { MatSnackBar } from '@angular/material';
import { AngularFirestore, AngularFirestoreDocument } from '@angular/fire/firestore';
import { MatDialog } from '@angular/material/dialog';
import { ConfirmationDialogComponent } from '../shared/confirmation-dialog/confirmation-dialog.component';
import * as firebase from 'firebase';
@Injectable({
    providedIn: 'root'
})
export class HomeService {

    loggedIn = new BehaviorSubject<boolean>(false);
    constructor(private snackBar: MatSnackBar, private db: AngularFirestore, public dialog: MatDialog) { }

    // Delete already exist file in the firestore
    deleteFile(filename, uid) {
        return new Promise((resolve) => {
            const deletFileQuery = this.db.collection('uploadFiles').ref.where('id', '==', uid).where('name', '==', filename);
            deletFileQuery.get().then((querySnapshot) => {
                if (querySnapshot.empty) {
                    resolve(false);
                    return;
                } else {
                    this.showDeleteConfirm().then((result) => {
                        if (!result) { return; }
                        querySnapshot.forEach((doc) => {
                            doc.ref.delete();
                            resolve(true);
                        });
                    });
                }

            });
        });
    }

    // Update Data to the firestore based on the logged in user
    updateDb(uid, delim, fileName) {
        return new Promise((resolve) => {
            this.db.collection('uploadFiles').add({
                name: fileName,
                id: uid,
                delimiter: delim.name,
                date: new Date()
            }).then((data) => {
                data.onSnapshot((snapshot) => {
                    this.snackBar.open('File Uploaded Succesfully!', 'close', { duration: 2000 });
                    resolve(snapshot);
                });
            }).catch((err) => {
                this.snackBar.open(err.message, 'close', { duration: 2000 });
                resolve(false);
            });
        });
    }

    // show confirmation dialog box
    showDeleteConfirm() {
        return new Promise((resolve) => {
            const dialogRef = this.dialog.open(ConfirmationDialogComponent);
            dialogRef.afterClosed().subscribe(result => {
                resolve(result);
            });
        });
    }


    // update Download URL to the File uploaded
    updateDownloadURL(url, id, docId) {
        const documentId = docId;
        const downloadURLref: AngularFirestoreDocument = this.db.doc(`downloadurls/${id}`);
        let obj = {};
        obj[documentId] = url;
        downloadURLref.update({
            url:  firebase.firestore.FieldValue.arrayUnion(obj)
        });
        obj = {};
    }
}
