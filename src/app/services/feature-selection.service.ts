import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material';
import { HttpClient } from '@angular/common/http';
import { URL, PORT } from '../constant/app.constants';
import { BehaviorSubject } from 'rxjs';
@Injectable({
    providedIn: 'root'
})

export class FeatureSelectionService {
    dragAndDrop: BehaviorSubject<Object> = new BehaviorSubject({})
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

    executeLinearRegressionAlgorithm(obj) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/models/regressor/linear`, obj).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }

    executeRandomForestRegressionAlgorithm(obj) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/models/regressor/random-forest`, obj).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }

    executeKNNRegressionAlgorithm(obj) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/models/regressor/knn`, obj).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }

    executeLogisticClassifierAlgorithm(obj) {
        console.log(obj);
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/models/classifier/logistic`, obj).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }

    executeRandomForestClassifierAlgorithm(obj) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/models/classifier/random-forest`, obj).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }
    
    executeGradientBoostClassifierAlgorithm(obj) {
        return new Promise((resolve) => {
            this.http.post(`${this.url}:${this.port}/api/models/classifier/gb`, obj).subscribe((res) => {
                console.log('response api', res);
                resolve(res);
            });
        });
    }
}
