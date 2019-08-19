import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material';
import { HttpClient } from '@angular/common/http';
import { URL, PORT } from '../constant/app.constants';
@Injectable({
    providedIn: 'root'
})
export class FeatureSelectionService {
    user;
    url = URL;
    port = PORT;
    fileListForUser: any[] = [];
    constructor(private snackBar: MatSnackBar, private http: HttpClient) {
        this.user = JSON.parse(localStorage.getItem('user'));
    }
 

    loadChart(data) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/chart`, data).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }

    getMissingValues(data) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/imputedValues`, data).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }
}
