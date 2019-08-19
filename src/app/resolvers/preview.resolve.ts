import { Injectable } from '@angular/core';

import { Resolve } from '@angular/router';

import { Observable } from 'rxjs';
import { of } from 'rxjs';
import { delay } from 'rxjs/operators';
import { PreviewService } from '../services/preview.service';


@Injectable()
export class PreviewResolver implements Resolve<Observable<any>> {
    constructor(private previewService: PreviewService) { }

    resolve() {
        return this.previewService.getFilesForUsers();
    }
}