'use strict';

import {
    Block,
    Div,
    Table,
    Tools
} from '@src/classes';

export interface TableHeadCellSearchData {
    text: string;
}

export interface TableHeadCellSortData {
    sort: number;
}

export class TableHeadCell extends Block {

    private labelContainer: Div;
    private searchButton: Div;
    private label: Div;
    private sort: Div;
    private searchContainer: Div;
    private searchInput: Block;
    private searchTimeout: any;

    static readonly SORT_DISABLED = 0;
    static readonly SORT_ASCENDING = 1;
    static readonly SORT_DESCENDING = 2;

    constructor(public column: string, private headRow: Block) {
    
        super('th', {}, headRow);

        this.labelContainer = new Div('label-container', this);
        this.searchButton = new Div('search-button', this.labelContainer);
        this.label = new Div('label', this.labelContainer).write(column);
        this.sort = new Div('sort', this.labelContainer);
        
        this.searchContainer = new Div('search-container', this);

        this.searchInput = new Block('input', {
            type: 'text',
            placeholder: 'Search'
        }, this.searchContainer);
        
        this.setData('sort', Table.SORT_DISABLED);

        this.onNative('click', this.onSort.bind(this));
        this.searchButton.onNative('click', this.onSearchButton.bind(this));
        this.searchInput.onNative('input', this.onSearch.bind(this));

        this.searchInput.onNative('click', (event: Event) => { event.stopPropagation(); });
    }

    /*
    **
    **
    */
    private async onSearchButton(event: Event) : Promise<void> {

        event.stopPropagation();

        this.searchInput.element.value = '';

        if (parseInt(this.getData('search-displayed')) === 1)
            this.setData('search-displayed', 0);
        else
            this.setData('search-displayed', 1);
        
        this.emitSearchData();

        await Tools.sleep(5);
        this.searchInput.element.focus();
    }

    /*
    **
    **
    */
    private onSearch() : void {
        
        clearTimeout(this.searchTimeout);

        this.searchTimeout = setTimeout(this.emitSearchData.bind(this), 100);
    }

    /*
    **
    **
    */
    private onSort() : void {

        let oldSort = parseInt(this.getData('sort'));
        let newSort = Table.SORT_DISABLED;

        if (oldSort === Table.SORT_DISABLED)
            newSort = Table.SORT_ASCENDING;
        else if (oldSort === Table.SORT_ASCENDING)
            newSort = Table.SORT_DESCENDING;

        this.setData('sort', newSort);
        
        this.emitSortData();
    }

    /*
    **
    **
    */
    private emitSearchData() : void {

        this.emit('search-data', {
            text: this.searchInput.element.value
        });

        this.emit('resize');
    }

    /*
    **
    **
    */
    private emitSortData() : void {

        this.emit('sort-data', {
            sort: parseInt(this.getData('sort'))
        });

        this.emit('resize');
    }

    /*
    **
    **
    */
    public resetSort() : void {

        this.setData('sort', Table.SORT_DISABLED);

        this.emit('resize');
    }
}