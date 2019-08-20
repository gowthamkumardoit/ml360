import { Component, OnInit, Input } from '@angular/core';
import { CdkDragDrop, moveItemInArray, transferArrayItem } from '@angular/cdk/drag-drop';
import { FeatureSelectionService } from 'src/app/services/feature-selection.service';

@Component({
  selector: 'app-drag-drop',
  templateUrl: './drag-drop.component.html',
  styleUrls: ['./drag-drop.component.scss']
})
export class DragDropComponent implements OnInit {
  original_columns: any[] =[];
  feature_columns: any;
  constructor(private featureSelectionService: FeatureSelectionService) { }

  ngOnInit() {
    this.featureSelectionService.dragAndDrop.subscribe((data: any) => {
      console.table('data', data);
      this.original_columns = [];
      if (data && data.original) {
        data.original.forEach(ele => {
          this.original_columns.push(ele.column);
        })
        this.feature_columns = data.featured;
      }
    });
    console.log('----',  this.featureSelectionService.dragAndDrop);
  }




  drop(event: CdkDragDrop<string[]>) {
    if (event.previousContainer === event.container) {
      moveItemInArray(event.container.data, event.previousIndex, event.currentIndex);
    } else {
      transferArrayItem(event.previousContainer.data,
        event.container.data,
        event.previousIndex,
        event.currentIndex);
    }
  }

}
