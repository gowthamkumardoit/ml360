import { MatToolbarModule, MatButtonModule, MatTabsModule } from '@angular/material'
import { NgModule } from '@angular/core';

@NgModule({
  imports: [MatToolbarModule, MatButtonModule, MatTabsModule
  ],
  exports: [MatToolbarModule, MatButtonModule, MatTabsModule]
  
})
export class MaterialModule { }
