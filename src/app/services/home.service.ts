import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { MatSnackBar } from '@angular/material';
import { AngularFirestore } from '@angular/fire/firestore';
import { MatDialog } from '@angular/material/dialog';
import { ConfirmationDialogComponent } from '../shared/confirmation-dialog/confirmation-dialog.component';
@Injectable({
    providedIn: 'root'
})
export class HomeService {
    loggedIn = new BehaviorSubject<boolean>(false);
    constructor(private snackBar: MatSnackBar, private db: AngularFirestore, public dialog: MatDialog) { }

    // Delete already exist file in the firestore
    deleteFile(filename) {
        return new Promise((resolve) => {
            const deletFileQuery = this.db.collection('uploadFiles').ref.where('name', '==', filename);
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
                this.snackBar.open('File Uploaded Succesfully!', 'close', { duration: 2000 });
                resolve(true);
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
}